"""Distributed borrow checking with epoch- and fencing-token leases.

Rust's borrow checker proves exclusivity statically.  Across a network we
cannot stop a partitioned node from writing, so the runtime maps the two Rust
borrow modes onto *leases*:

* ``exclusive`` -- one owner, equivalent to ``&mut T``;
* ``shared``    -- many owners, equivalent to ``&T``; writes are rejected.

Every grant carries a monotonically increasing **fencing token** per key and
the current cluster **epoch**.  A write must present ``(owner, epoch, fencing)``
and the authority rejects anything that is not the newest grant.  That is the
property that prevents split-brain writes after a migration or a partition:
even if an old node still believes it holds the lease, its token is stale and
the store refuses the write.

The manager is the single authority for a key set.  In a real cluster it runs
on one coordinator (or one authority per key shard); :class:`LeaseClient`
speaks to it over the control-plane transport.  The in-memory cluster in
``distributed.cluster`` uses the same request/reply shape.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from distributed.diagnostics import (
    DistributedError,
    lease_conflict,
    lease_expired,
    stale_fence,
)
from distributed.transmute import register_type

SHARED: Final = "shared"
EXCLUSIVE: Final = "exclusive"
MODES: Final = (SHARED, EXCLUSIVE)


@dataclass(frozen=True)
class LeaseToken:
    """A grant that can be validated by any holder of the authority state."""

    key: str
    mode: str
    owner: str
    epoch: int
    fencing: int
    expires_at: float
    authority: str = "lease-authority"

    @property
    def is_shared(self) -> bool:
        return self.mode == SHARED

    def is_expired(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) >= self.expires_at

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "mode": self.mode,
            "owner": self.owner,
            "epoch": self.epoch,
            "fencing": self.fencing,
            "expires_at": self.expires_at,
            "authority": self.authority,
        }


register_type(LeaseToken, name="LeaseToken")


def _check_mode(mode: str) -> str:
    if mode not in MODES:
        raise DistributedError(
            f"unknown lease mode {mode!r}",
            code="E003",
            note="LeaseModeError",
            help_text="use 'shared' or 'exclusive'",
        )
    return mode


class LeaseManager:
    """The authoritative lease table for a key space."""

    def __init__(
        self,
        *,
        authority: str = "lease-authority",
        ttl: float = 30.0,
        clock: Callable[[], float] | None = None,
        epoch: int = 0,
        max_leases: int = 100_000,
    ) -> None:
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        self.authority = authority
        self.ttl = float(ttl)
        self.clock = clock or time.time
        self.epoch = int(epoch)
        self.max_leases = max_leases
        self._leases: dict[str, dict[str, LeaseToken]] = {}
        self._fencing: dict[str, int] = {}
        self._lock = threading.RLock()
        # Statistics used by diagnostics and tests.
        self.grants = 0
        self.renewals = 0
        self.expirations = 0
        self.rejections = 0

    # -- epoch / time -----------------------------------------------------
    @property
    def now(self) -> float:
        return float(self.clock())

    def bump_epoch(self) -> int:
        """Start a new cluster term and revoke every lease from the old one.

        Surviving nodes re-acquire their leases under the new epoch; tokens
        from the previous term are rejected on their epoch before ownership is
        even considered.  Fencing counters are preserved so a re-granted key
        never reuses a number.
        """
        with self._lock:
            self.epoch += 1
            self._leases.clear()
            return self.epoch

    def purge_expired(self) -> int:
        with self._lock:
            now = self.now
            removed = 0
            for key in list(self._leases):
                owners = self._leases[key]
                for owner, token in list(owners.items()):
                    if now >= token.expires_at:
                        del owners[owner]
                        removed += 1
                if not owners:
                    del self._leases[key]
            self.expirations += removed
            return removed

    # -- core operations --------------------------------------------------
    def acquire(
        self,
        key: str,
        owner: str,
        mode: str = EXCLUSIVE,
        *,
        ttl: float | None = None,
    ) -> LeaseToken:
        mode = _check_mode(mode)
        with self._lock:
            self.purge_expired()
            owners = self._leases.setdefault(key, {})
            existing = owners.get(owner)
            if existing is not None and existing.mode == mode:
                return self._renew_locked(existing, ttl)
            for other_owner, other in owners.items():
                if other_owner == owner:
                    continue
                if mode == EXCLUSIVE or other.mode == EXCLUSIVE:
                    self.rejections += 1
                    raise lease_conflict(
                        f"lease {key!r} is held {other.mode} by {other_owner!r}",
                        reason=(
                            f"requested {mode} by {owner!r}; "
                            f"conflicts with {other.mode} by {other_owner!r}"
                        ),
                        context={
                            "key": key,
                            "requested_by": owner,
                            "requested_mode": mode,
                            "holder": other_owner,
                            "holder_mode": other.mode,
                        },
                    )
            if existing is not None:
                # Mode upgrade/downgrade; the conflict scan above proved it is
                # safe.  Upgrade keeps the same owner but bumps the fence.
                del owners[owner]
            return self._grant_locked(key, owner, mode, ttl)

    def renew(self, token: LeaseToken, *, ttl: float | None = None) -> LeaseToken:
        with self._lock:
            self._validate_locked(token, write=False, allow_expired=False)
            return self._renew_locked(token, ttl)

    def release(self, token: LeaseToken) -> None:
        with self._lock:
            self._validate_locked(token, write=False, allow_expired=True)
            owners = self._leases.get(token.key, {})
            current = owners.get(token.owner)
            if current is None:
                return
            if current.fencing != token.fencing:
                self.rejections += 1
                raise stale_fence(
                    f"cannot release {token.key!r}: lease was superseded",
                    reason=(f"token {token.fencing} is older than {current.fencing}"),
                    context=token.describe(),
                )
            del owners[token.owner]
            if not owners:
                self._leases.pop(token.key, None)

    def transfer(
        self,
        key: str,
        from_owner: str,
        to_owner: str,
        *,
        mode: str = EXCLUSIVE,
        ttl: float | None = None,
    ) -> LeaseToken:
        """Atomically revoke ``from_owner`` and grant ``to_owner``.

        Migration uses this to cut the source node off before the target
        starts.  The new grant always has a higher fencing token, so a late
        write from the old owner is rejected.
        """
        mode = _check_mode(mode)
        with self._lock:
            self.purge_expired()
            owners = self._leases.setdefault(key, {})
            for other_owner, other in owners.items():
                if other_owner in {from_owner, to_owner}:
                    continue
                if mode == EXCLUSIVE or other.mode == EXCLUSIVE:
                    self.rejections += 1
                    raise lease_conflict(
                        f"cannot transfer {key!r}: held by {other_owner!r}",
                        context={"key": key, "holder": other_owner},
                    )
            owners.pop(from_owner, None)
            owners.pop(to_owner, None)
            token = self._grant_locked(key, to_owner, mode, ttl)
            return token

    def validate(
        self,
        token: LeaseToken,
        *,
        write: bool = False,
        allow_expired: bool = False,
    ) -> LeaseToken:
        with self._lock:
            return self._validate_locked(
                token, write=write, allow_expired=allow_expired
            )

    def holders(self, key: str) -> dict[str, LeaseToken]:
        with self._lock:
            self.purge_expired()
            return dict(self._leases.get(key, {}))

    def fencing(self, key: str) -> int:
        with self._lock:
            return self._fencing.get(key, 0)

    # -- internals --------------------------------------------------------
    def _grant_locked(
        self, key: str, owner: str, mode: str, ttl: float | None
    ) -> LeaseToken:
        if len(self._leases) >= self.max_leases and key not in self._leases:
            raise DistributedError(
                f"lease table is full ({self.max_leases} keys)",
                note="LeaseTableFullError",
                help_text="release unused keys or shard the authority",
            )
        fencing = self._fencing.get(key, 0) + 1
        self._fencing[key] = fencing
        duration = self.ttl if ttl is None else float(ttl)
        if duration <= 0:
            raise DistributedError(
                "lease ttl must be positive",
                code="E003",
                note="LeaseModeError",
            )
        token = LeaseToken(
            key=key,
            mode=mode,
            owner=owner,
            epoch=self.epoch,
            fencing=fencing,
            expires_at=self.now + duration,
            authority=self.authority,
        )
        self._leases.setdefault(key, {})[owner] = token
        self.grants += 1
        return token

    def _renew_locked(self, token: LeaseToken, ttl: float | None) -> LeaseToken:
        current = self._leases.get(token.key, {}).get(token.owner)
        if current is None or current.fencing != token.fencing:
            raise stale_fence(
                f"lease {token.key!r} is no longer held by {token.owner!r}",
                context=token.describe(),
            )
        duration = self.ttl if ttl is None else float(ttl)
        renewed = LeaseToken(
            key=current.key,
            mode=current.mode,
            owner=current.owner,
            epoch=self.epoch,
            fencing=current.fencing,
            expires_at=self.now + duration,
            authority=current.authority,
        )
        self._leases[current.key][current.owner] = renewed
        self.renewals += 1
        return renewed

    def _validate_locked(
        self,
        token: LeaseToken,
        *,
        write: bool,
        allow_expired: bool,
    ) -> LeaseToken:
        if token.epoch != self.epoch:
            self.rejections += 1
            raise stale_fence(
                f"lease {token.key!r} belongs to epoch {token.epoch}; "
                f"authority is at epoch {self.epoch}",
                reason="a cluster term change revoked every earlier lease",
                context=token.describe(),
            )
        current = self._leases.get(token.key, {}).get(token.owner)
        if current is None:
            self.rejections += 1
            raise stale_fence(
                f"lease {token.key!r} is not held by {token.owner!r}",
                context=token.describe(),
            )
        if current.fencing != token.fencing:
            self.rejections += 1
            raise stale_fence(
                f"stale fencing token for {token.key!r}: "
                f"{token.fencing} < {current.fencing}",
                reason=f"the current grant is fencing {current.fencing}",
                context=token.describe(),
            )
        if write and current.mode == SHARED:
            self.rejections += 1
            raise lease_conflict(
                f"cannot write {token.key!r} through a shared lease",
                reason="shared leases map to &T, not &mut T",
                context=token.describe(),
            )
        if not allow_expired and self.now >= current.expires_at:
            self.rejections += 1
            raise lease_expired(
                f"lease {token.key!r} expired at {current.expires_at:.3f}",
                help_text="renew the lease before the TTL elapses",
                context=token.describe(),
            )
        return current

    # -- snapshot / restore ----------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "authority": self.authority,
                "ttl": self.ttl,
                "epoch": self.epoch,
                "fencing": dict(self._fencing),
                "leases": {
                    key: {owner: token for owner, token in owners.items()}
                    for key, owners in self._leases.items()
                },
                "stats": {
                    "grants": self.grants,
                    "renewals": self.renewals,
                    "expirations": self.expirations,
                    "rejections": self.rejections,
                },
            }

    def restore(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.authority = payload.get("authority", self.authority)
            self.ttl = float(payload.get("ttl", self.ttl))
            self.epoch = int(payload.get("epoch", self.epoch))
            self._fencing = {
                str(key): int(value)
                for key, value in payload.get("fencing", {}).items()
            }
            leases: dict[str, dict[str, LeaseToken]] = {}
            for key, owners in payload.get("leases", {}).items():
                leases[str(key)] = {
                    str(owner): token for owner, token in owners.items()
                }
            self._leases = leases
            stats = payload.get("stats", {})
            self.grants = int(stats.get("grants", self.grants))
            self.renewals = int(stats.get("renewals", self.renewals))
            self.expirations = int(stats.get("expirations", self.expirations))
            self.rejections = int(stats.get("rejections", self.rejections))

    # -- request/reply surface -------------------------------------------
    def handle_request(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle one control-plane lease request.

        This is the exact surface :class:`LeaseClient` speaks, so the same
        handler can be mounted behind an in-memory or TCP transport.
        """
        try:
            if op == "acquire":
                token = self.acquire(
                    payload["key"],
                    payload["owner"],
                    payload.get("mode", EXCLUSIVE),
                    ttl=payload.get("ttl"),
                )
                return {"ok": True, "token": token}
            if op == "renew":
                token = self.renew(payload["token"], ttl=payload.get("ttl"))
                return {"ok": True, "token": token}
            if op == "release":
                self.release(payload["token"])
                return {"ok": True}
            if op == "transfer":
                token = self.transfer(
                    payload["key"],
                    payload["from_owner"],
                    payload["to_owner"],
                    mode=payload.get("mode", EXCLUSIVE),
                    ttl=payload.get("ttl"),
                )
                return {"ok": True, "token": token}
            if op == "validate":
                token = self.validate(
                    payload["token"],
                    write=bool(payload.get("write")),
                    allow_expired=bool(payload.get("allow_expired")),
                )
                return {"ok": True, "token": token}
            if op == "holders":
                return {
                    "ok": True,
                    "holders": self.holders(payload["key"]),
                }
            if op == "fencing":
                return {"ok": True, "fencing": self.fencing(payload["key"])}
            if op == "snapshot":
                return {"ok": True, "snapshot": self.snapshot()}
            raise DistributedError(
                f"unknown lease operation {op!r}",
                code="E003",
                note="LeaseProtocolError",
            )
        except DistributedError as error:
            return {"ok": False, "error": error}


class LeaseGuard:
    """Context manager that acquires a lease and releases it on exit."""

    def __init__(
        self,
        manager: LeaseManager,
        key: str,
        owner: str,
        mode: str = EXCLUSIVE,
        *,
        ttl: float | None = None,
    ) -> None:
        self.manager = manager
        self.key = key
        self.owner = owner
        self.mode = mode
        self.ttl = ttl
        self.token: LeaseToken | None = None

    def __enter__(self) -> LeaseToken:
        self.token = self.manager.acquire(self.key, self.owner, self.mode, ttl=self.ttl)
        return self.token

    def __exit__(self, *exc_info: object) -> None:
        if self.token is not None:
            try:
                self.manager.release(self.token)
            except DistributedError:
                # The lease may have been transferred by a migration; do not
                # mask the original exception.
                pass

    def renew(self, *, ttl: float | None = None) -> LeaseToken:
        if self.token is None:
            raise DistributedError(
                "lease guard has not been entered",
                code="E003",
                note="LeaseModeError",
            )
        self.token = self.manager.renew(self.token, ttl=ttl)
        return self.token

    def validate_write(self) -> LeaseToken:
        if self.token is None:
            raise DistributedError(
                "lease guard has not been entered",
                code="E003",
                note="LeaseModeError",
            )
        return self.manager.validate(self.token, write=True)


class LeaseClient:
    """A client-side view of a remote lease authority."""

    def __init__(self, call: Callable[[str, dict[str, Any]], dict[str, Any]]):
        self._call = call

    def _request(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        reply = self._call(op, payload)
        if not reply.get("ok", False):
            error = reply.get("error")
            if isinstance(error, DistributedError):
                raise error
            raise DistributedError(
                f"lease authority rejected {op}: {error}",
                note="LeaseProtocolError",
            )
        return reply

    def acquire(
        self,
        key: str,
        owner: str,
        mode: str = EXCLUSIVE,
        *,
        ttl: float | None = None,
    ) -> LeaseToken:
        return self._request(
            "acquire", {"key": key, "owner": owner, "mode": mode, "ttl": ttl}
        )["token"]

    def renew(self, token: LeaseToken, *, ttl: float | None = None) -> LeaseToken:
        return self._request("renew", {"token": token, "ttl": ttl})["token"]

    def release(self, token: LeaseToken) -> None:
        self._request("release", {"token": token})

    def transfer(
        self,
        key: str,
        from_owner: str,
        to_owner: str,
        *,
        mode: str = EXCLUSIVE,
        ttl: float | None = None,
    ) -> LeaseToken:
        return self._request(
            "transfer",
            {
                "key": key,
                "from_owner": from_owner,
                "to_owner": to_owner,
                "mode": mode,
                "ttl": ttl,
            },
        )["token"]

    def validate(
        self,
        token: LeaseToken,
        *,
        write: bool = False,
        allow_expired: bool = False,
    ) -> LeaseToken:
        return self._request(
            "validate",
            {
                "token": token,
                "write": write,
                "allow_expired": allow_expired,
            },
        )["token"]

    def holders(self, key: str) -> dict[str, LeaseToken]:
        return self._request("holders", {"key": key})["holders"]

    def fencing(self, key: str) -> int:
        return int(self._request("fencing", {"key": key})["fencing"])


__all__ = [
    "EXCLUSIVE",
    "MODES",
    "SHARED",
    "LeaseClient",
    "LeaseGuard",
    "LeaseManager",
    "LeaseToken",
]

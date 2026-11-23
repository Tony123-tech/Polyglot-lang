# Distributed cluster and location-transparent actors

`distributed/frontier.py` extends the existing `distributed/cluster.py` model with
membership discovery, session-typed routing, and split-brain decisions. It reuses
`Cluster`, `LeaseManager`, `wire.encode_frame`/`decode_frame`, and the
`DistributedError` factories, so the module adds behaviour without a parallel
transport.

## Discovery and heartbeats

```python
from distributed.cluster import Cluster
from distributed.frontier import Heartbeat, cluster_discover_and_heartbeat

cluster = Cluster()
view = cluster_discover_and_heartbeat(
    cluster,
    [Heartbeat("edge-1", 1, 100), Heartbeat("edge-2", 1, 120)],
    now_ms=150,
)
view.members  # ('edge-1', 'edge-2')
view.discovered  # ('edge-2',)
view.quorum  # 2
```

Rules:

- Heartbeats are processed in `(node, seq)` order, so a batch is deterministic.
- An unknown sender is discovered and joined to the cluster.
- A repeated or older `seq` from the same node is ignored and reported in `stale`.
- A timestamp that moves backwards is `E001`; it is a bug in the sender, not
  clock skew to be smoothed over.
- A member silent for longer than `ttl_ms` is evicted: it is marked dead in
  `membership`, added to `cluster.dead`, and returned in `evicted`.

`ttl_ms` history is kept on the cluster (`cluster._frontier_seq`,
`cluster._frontier_sent`) so two calls in a row behave as one timeline.

## Session-typed routing

```python
from distributed.frontier import cluster_route_session_typed_rpc

rpc = cluster_route_session_typed_rpc(
    cluster,
    (("client", "Request"), ("server", "Response")),
    route_key="account-42",
    direction="client",
    message_type="Request",
    payload={"amount": 7},
)
rpc.target  # one live node, chosen by hash of route_key
rpc.session_id  # stable id for this route/direction/type/epoch
rpc.remaining  # (('server', 'Response'),)
rpc.wire_version  # 1
```

- The first protocol step must match `direction`/`message_type`, otherwise
  `E001`.
- `schema_version` outside `supported_versions` is `E007`.
- The payload is normalised with sorted JSON keys, encoded with
  `encode_frame`, and decoded again; a round-trip mismatch is `E004`.
- A cluster with no live member raises `E006`.

## Partitions and split brain

```python
report = cluster_handle_partition_and_split_brain(cluster, [["a", "b"], ["c"]])
report.decision  # 'majority'
report.epoch  # bumped to the next fencing epoch
report.fences  # {'c': <epoch>}
```

- A group that reaches `total // 2 + 1` nodes is writable and clears its dead
  set; the other groups are marked dead and fenced at the new epoch.
- When no group reaches quorum the cluster freezes: every partition stays
  read-only and the epoch does not move. Freezing is the safe answer; a
  minority that keeps accepting writes is exactly the split-brain this avoids.
- Unknown nodes, overlapping partitions, and an empty cluster are `E001`/`E002`.

## Boundaries

This is a single-process model. Membership state, leases and codecs are in
memory; there is no socket layer, no gossip, and no persistence. What it
demonstrates is the decision logic: who is alive, which partition may write,
which session turn may send, and how a stale epoch is refused.

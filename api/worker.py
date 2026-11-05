"""
A: in-process queue worker.

For low-traffic deployments, we process requests sequentially via an
asyncio.Queue and a single worker coroutine. This serialises compilation
so that concurrent requests do not race on the cargo cache.

Redis backend is a stub for future multi-worker setups.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from api.config import Config
from api.executor import (
    DockerExecutor,
    SubprocessExecutor,
    make_executor,
)


@dataclass
class Job:
    source: str
    backend: str
    trusted: bool
    future: asyncio.Future


class InProcessQueue:
    """A: single worker coroutine, asyncio.Queue for requests."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._executor: SubprocessExecutor | DockerExecutor = make_executor(cfg)
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def submit(
        self, source: str, backend: str, trusted: bool,
    ) -> tuple[int, str, str]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put(Job(source, backend, trusted, fut))
        return await fut

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                result = await self._executor.run(
                    job.source, job.backend, job.trusted
                )
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as e:  # noqa: BLE001
                if not job.future.done():
                    job.future.set_exception(e)
            finally:
                self._queue.task_done()


def make_queue(cfg: Config) -> InProcessQueue:
    # Redis queue is a future addition; always return in-process for now.
    return InProcessQueue(cfg)

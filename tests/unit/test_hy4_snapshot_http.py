import asyncio
import os
import sys
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/apps/server"))
from shelf_server import create_app
from serverlib.compute_client import AsyncComputeClient


class Compute:
    def __init__(self):
        self.result = {"event": "snapshot", "status": "ok", "bytes": 4096}
        self.calls = 0

    async def snapshot(self):
        self.calls += 1
        return self.result

    async def shutdown(self):
        pass


async def check():
    compute = Compute()
    app = create_app(None, compute, {"expert_parallel": 3})
    async with TestClient(TestServer(app)) as client:
        os.environ.pop("SHELF_HY4_SNAPSHOT", None)
        response = await client.post("/admin/snapshot")
        assert response.status == 404 and compute.calls == 0
        os.environ["SHELF_HY4_SNAPSHOT"] = "1"
        response = await client.post("/admin/snapshot")
        assert response.status == 200 and (await response.json())["bytes"] == 4096
        compute.result = {"event": "snapshot", "status": "error", "message": "失敗"}
        response = await client.post("/admin/snapshot")
        assert response.status == 503 and (await response.json())["message"] == "失敗"
    other = create_app(None, compute, {})
    async with TestClient(TestServer(other)) as client:
        response = await client.post("/admin/snapshot")
        assert response.status == 404
    os.environ.pop("SHELF_HY4_SNAPSHOT", None)
    protocol = AsyncComputeClient([], max_inflight=1)
    sent = []

    async def send(payload):
        sent.append(payload)
        await protocol._control.put({"event": "snapshot", "status": "ok"})

    protocol._send = send
    await protocol._semaphore.acquire()
    task = asyncio.create_task(protocol.snapshot())
    await asyncio.sleep(0)
    assert not sent
    protocol._semaphore.release()
    assert (await task)["status"] == "ok" and sent == [{"op": "snapshot"}]


if __name__ == "__main__":
    asyncio.run(check())
    print("状態退避 HTTP の検証に成功しました")

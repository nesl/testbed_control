import asyncio
import base64
import json
import os
import time

import websockets
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


WS_URL = "ws://127.0.0.1:12345"
API_SECRET_KEY = "cDdzYXNkbXM5d2V2a3EwaTJ0Z2tocHRlNjE2NWs5ODY="

CLIENT_ID = 1
CCT = 3200

# Official range: [0, 1000], where 1000 = 100%.
START_INTENSITY = 5  # 0.1%.
INTENSITY_DELTA = 5  # 1 + 9 = 10, which is 1%.

_last_request_id = 0


def generate_token(secret_key: str) -> str:
    iv = os.urandom(12)
    encryptor = Cipher(
        algorithms.AES(base64.b64decode(secret_key)),
        modes.GCM(iv),
        backend=default_backend(),
    ).encryptor()
    ciphertext = encryptor.update(str(int(time.time())).encode()) + encryptor.finalize()
    return base64.b64encode(iv + encryptor.tag + ciphertext).decode()


def next_request_id() -> int:
    global _last_request_id
    request_id = int(time.time() * 1000)
    if request_id <= _last_request_id:
        request_id = _last_request_id + 1
    _last_request_id = request_id
    return request_id


async def recv_response(ws, request_id):
    while True:
        data = json.loads(await ws.recv())
        if data.get("type") == "event":
            print("EVENT:", json.dumps(data, indent=2))
            continue
        if data.get("type") == "response" and data.get("request_id") == request_id:
            print("RECV:", json.dumps(data, indent=2))
            return data
        print("IGNORED:", json.dumps(data, indent=2))


async def send_v2(ws, action, node_id=None, args=None):
    request_id = next_request_id()
    request = {
        "version": 2,
        "type": "request",
        "client_id": CLIENT_ID,
        "request_id": request_id,
        "action": action,
        "token": generate_token(API_SECRET_KEY),
    }
    if node_id is not None:
        request["node_id"] = node_id
    if args is not None:
        request["args"] = args

    print("SEND:", json.dumps(request, indent=2))
    await ws.send(json.dumps(request))
    return await recv_response(ws, request_id)


def ensure_ok(resp, action):
    if resp.get("code") != 0:
        raise RuntimeError(f"{action} failed: {json.dumps(resp, indent=2)}")
    return resp


def extract_devices(resp):
    data = resp.get("data", resp)
    return data if isinstance(data, list) else []


async def main():
    async with websockets.connect(WS_URL) as ws:
        ensure_ok(await send_v2(ws, "get_protocol_versions"), "get_protocol_versions")

        devices = extract_devices(ensure_ok(await send_v2(ws, "get_fixture_list"), "get_fixture_list"))
        if not devices:
            devices = extract_devices(ensure_ok(await send_v2(ws, "get_device_list"), "get_device_list"))
        if not devices:
            raise RuntimeError("No light found.")

        light = devices[0]
        node_id = light["node_id"]
        print(f"\nUsing light: {light.get('name')} ({node_id})")

        ensure_ok(await send_v2(ws, "set_sleep", node_id=node_id, args={"sleep": False}), "set_sleep")
        await asyncio.sleep(1)

        # Set CCT mode first, then set the target brightness.
        ensure_ok(
            await send_v2(ws, "set_cct", node_id=node_id, args={"cct": CCT}),
            "set_cct",
        )
        print(f"\nSetting brightness to 0.1% ({START_INTENSITY}/1000)")
        ensure_ok(
            await send_v2(ws, "set_intensity", node_id=node_id, args={"intensity": START_INTENSITY}),
            "set_intensity",
        )
        readback = ensure_ok(await send_v2(ws, "get_intensity", node_id=node_id), "get_intensity")
        print(f"Confirmed intensity: {readback.get('data')} / 1000")
        await asyncio.sleep(4)

        print(f"\nIncreasing brightness by {INTENSITY_DELTA} to 1%")
        ensure_ok(
            await send_v2(ws, "increase_intensity", node_id=node_id, args={"delta": INTENSITY_DELTA}),
            "increase_intensity",
        )
        readback = ensure_ok(await send_v2(ws, "get_intensity", node_id=node_id), "get_intensity")
        print(f"Confirmed intensity: {readback.get('data')} / 1000")
        await asyncio.sleep(4)

        cct_state = ensure_ok(await send_v2(ws, "get_cct", node_id=node_id), "get_cct")
        print("\nFinal CCT state:", json.dumps(cct_state.get("data"), indent=2))


if __name__ == "__main__":
    asyncio.run(main())

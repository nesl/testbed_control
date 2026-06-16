import asyncio
import json
import time
import os
import base64

import websockets
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


# API secret key applied for
api_secret_key = '9veqiL0G0EUOviwzL1prPc0iGIGUJtbzSaPYQfgfyxM='


# Generate token using AES-256-GCM algorithm
def generate_token(secret_key: str) -> str:
    iv = os.urandom(12)
    encryptor = Cipher(algorithms.AES(base64.b64decode(secret_key)), modes.GCM(iv), backend=default_backend()).encryptor()
    # Must use the current timestamp, which will be used to calculate the token's expiration time, must be an integer
    now = int(time.time())
    ciphertext = encryptor.update(str(now).encode()) + encryptor.finalize()
    combined = iv + encryptor.tag + ciphertext
    return base64.b64encode(combined).decode()


async def websocket_client():
    uri = "ws://127.0.0.1:12345"

    async with websockets.connect(uri) as websocket:
        request_message = {
            "version": 2,
            "type": "request",
            "client_id": 1,
            "request_id": 123,
            "action": "get_protocol_versions",
            # Generate a new token for each request, otherwise the token will expire, the validity period is 10s
            "token": generate_token(api_secret_key),
        }

        await websocket.send(json.dumps(request_message))
        print(f"Sent: {request_message}")

        response = await websocket.recv()
        print(f"Received: {response}")


asyncio.run(websocket_client())
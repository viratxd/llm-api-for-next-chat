import json
import os
from fastapi import HTTPException
from curl_cffi import requests
from curl_cffi.requests import AsyncSession
from utility import color_print, get_user_agent


class Deepseek_Web_RE:
    base_url = "https://chat.deepseek.com"
    api_prefix = f"{base_url}/api/v0"
    headers = {"User-Agent": get_user_agent()}
    models = ["deepseek-v2.5"]

    def __init__(self):
        self.headers = self.headers | {
            "Authorization": f"Bearer {self.bearer_token}",
            "X-App-Version": self.app_version,
        }
        self.async_session = AsyncSession(headers=self.headers, impersonate="chrome", timeout=None)

    @property
    def bearer_token(self) -> str:
        return os.environ.get("DEEPSEEK_BEARER_TOKEN")

    @property
    def app_version(self) -> str:
        url = f"{self.base_url}/version.txt"
        response = requests.get(url, headers=self.headers)
        return response.text

    async def _get_session_id(self) -> str:
        url = f"{self.api_prefix}/chat_session/create"
        payload = {"agent": "chat"}
        response = await self.async_session.post(url, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=await response.atext())
        return response.json()["data"]["biz_data"]["id"]

    async def _delete_chat_session(self, chat_session_id: str):
        url = f"{self.api_prefix}/chat_session/delete"
        payload = {"chat_session_id": chat_session_id}
        response = await self.async_session.post(url, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=await response.atext())

    async def completions(self, message: str):
        if not self.bearer_token:
            raise HTTPException(status_code=401, detail="Please set the DEEPSEEK_BEARER_TOKEN environment variable")

        url = f"{self.api_prefix}/chat/completion"
        chat_session_id = await self._get_session_id()

        payload = {
            "challenge_response": None,
            "chat_session_id": chat_session_id,
            "parent_message_id": None,
            "prompt": message,
            "ref_file_ids": [],
            "search_enabled": False,
            "thinking_enabled": False,
        }
        response = await self.async_session.post(url, json=payload, stream=True)

        color_print(f"Deepseek Web Response Status Code: {response.status_code}", "blue")
        # error will still return 200 status code, but only json format
        if response.headers.get("content-type") == "application/json":
            error_json = json.loads(await response.atext())
            raise HTTPException(status_code=400, detail=error_json)

        await self._delete_chat_session(chat_session_id)
        return response

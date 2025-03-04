import json
import os
import base64
from fastapi import HTTPException
from curl_cffi import requests
from curl_cffi.requests import AsyncSession
from seleniumbase import SB
from utility import color_print
from .ds_wasm_pow import DS_WasmPow
from multiprocessing import Process, Queue

class Deepseek_Web_RE:
    base_url = "https://chat.deepseek.com"
    api_prefix = f"{base_url}/api/v0"
    models = ["deepseek-chat"]

    def __init__(self, cookies: dict, user_agent: str):
        self.cf_challenge_cookies = cookies
        self.user_agent = user_agent
        self.headers = {
            "user-agent": self.user_agent,
            "authorization": f"Bearer +mzX6SY48LgKHayFNCxQAfarRe8xqVKKxvfqKwi+oNheHF7fJAHGuen5qayACntq",
            "x-app-version": self.app_version,
        }
        self.async_session = AsyncSession(
            headers=self.headers, 
            impersonate="chrome", 
            timeout=None, 
            cookies=self.cf_challenge_cookies
        )
        self.ds_wasm_pow = DS_WasmPow("deepseek_web/sha3_wasm_bg.7b9ca65ddd.wasm")

    @staticmethod
    def _get_cf_challenge(queue: Queue, base_url: str):
        """Run in separate process to isolate event loops"""
        with SB(uc=True, headed=True, xvfb=True) as sb:
            sb.uc_gui_click_captcha()
            sb.activate_cdp_mode(base_url)  # Use the passed base_url
            cookies = {cookie.name: cookie.value for cookie in sb.cdp.get_all_cookies()}
            user_agent = sb.get_user_agent()
            queue.put((cookies, user_agent))

    @classmethod
    def create(cls):
        """Factory method to handle CF challenge in isolated process"""
        queue = Queue()
        p = Process(target=cls._get_cf_challenge, args=(queue, cls.base_url))  # Pass base_url
        p.start()
        p.join()
        
        cookies, user_agent = queue.get()
        if "cf_clearance" not in cookies:
            color_print("Failed to solve the Cloudflare challenge", "yellow")
            raise RuntimeError("Cloudflare challenge failed")
            
        return cls(cookies, user_agent)

    @property
    def bearer_token(self) -> str:
        return os.environ.get("DEEPSEEK_BEARER_TOKEN")

    @property
    def app_version(self) -> str:
        url = f"{self.base_url}/version.txt"
        response = requests.get(url, headers={"user-agent": self.user_agent}, cookies=self.cf_challenge_cookies)
        return response.text

    async def _create_pow_challenge(self) -> dict:
        url = f"{self.api_prefix}/chat/create_pow_challenge"
        payload = {"target_path": "/api/v0/chat/completion"}
        response = await self.async_session.post(url, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()["data"]["biz_data"]["challenge"]

    async def _get_ds_pow(self) -> str:
        challenge = await self._create_pow_challenge()

        answer = self.ds_wasm_pow.calculate_answer(
            challenge["challenge"], challenge["salt"], challenge["difficulty"], challenge["expire_at"]
        )
        if not answer:
            raise HTTPException(status_code=400, detail="Failed to solve the proof of work challenge")
        result = {
            "algorithm": "DeepSeekHashV1",
            "challenge": challenge["challenge"],
            "salt": challenge["salt"],
            "answer": answer,
            "signature": challenge["signature"],
            "target_path": "/api/v0/chat/completion",
        }
        return base64.b64encode(json.dumps(result).encode()).decode()

    async def _get_session_id(self) -> str:
        url = f"{self.api_prefix}/chat_session/create"
        payload = {"character_id": None}
        response = await self.async_session.post(url, json=payload)
        if response.status_code == 403:
            self.__init__()
            raise HTTPException(status_code=403, detail="Cloudflare challenge expired, please retry")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()["data"]["biz_data"]["id"]

    async def _delete_chat_session(self, chat_session_id: str):
        url = f"{self.api_prefix}/chat_session/delete"
        payload = {"chat_session_id": chat_session_id}
        response = await self.async_session.post(url, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

    async def completions(self, message: str):
        if not self.bearer_token:
            raise HTTPException(status_code=401, detail="Please set the DEEPSEEK_BEARER_TOKEN environment variable")

        chat_session_id = await self._get_session_id()

        url = f"{self.api_prefix}/chat/completion"
        headers = self.headers | {"x-ds-pow-response": await self._get_ds_pow()}
        payload = {
            "chat_session_id": chat_session_id,
            "parent_message_id": None,
            "prompt": message,
            "ref_file_ids": [],
            "search_enabled": False,
            "thinking_enabled": False,
        }
        response = await self.async_session.post(url, json=payload, headers=headers, stream=True)

        color_print(f"Deepseek Web Response Status Code: {response.status_code}", "blue")
        # error will still return 200 status code, but only json format
        if response.headers.get("content-type") == "application/json":
            error_json = json.loads(await response.atext())
            raise HTTPException(status_code=400, detail=error_json)

        await self._delete_chat_session(chat_session_id)
        return response
import json
import httpx
import os
from fastapi import HTTPException
from utility import color_print, get_user_agent


class Deepseek_Web_RE:
    base_url = "https://chat.deepseek.com"
    api_prefix = f"{base_url}/api/v0"
    headers = {"User-Agent": get_user_agent()}
    model_key_mapping = {"deepseek-chat": "deepseek_chat", "deepseek-code": "deepseek_code"}

    def __init__(self, async_client: httpx.AsyncClient = None):
        self.async_client = async_client or httpx.AsyncClient()
        self.headers = self.headers | {
            "Authorization": f"Bearer {self.bearer_token}",
            "X-App-Version": self.app_version,
        }

    @property
    def bearer_token(self) -> str:
        return os.environ.get("DEEPSEEK_BEARER_TOKEN")

    @property
    def app_version(self) -> str:
        url = f"{self.base_url}/version.txt"
        with httpx.Client() as client:
            response = client.get(url)
            return response.text

    async def _clear_context(self, model_class: str):
        url = f"{self.api_prefix}/chat/clear_context"
        payload = {"append_welcome_message": False, "model_class": model_class}
        response = await self.async_client.post(url, headers=self.headers, json=payload, timeout=None)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

    async def completions(self, message: str, model: str, temperature: float):
        if not self.bearer_token:
            raise HTTPException(status_code=401, detail="Please set the DEEPSEEK_BEARER_TOKEN environment variable")
        model_class = self.model_key_mapping.get(model, "deepseek_chat")

        await self._clear_context(model_class)

        url = f"{self.api_prefix}/chat/completions"
        payload = {
            "message": message,
            "model_class": model_class,
            "model_preference": None,
            "stream": True,
            "temperature": temperature,
        }
        json_payload = json.dumps(payload, ensure_ascii=False)
        req = self.async_client.build_request("POST", url, headers=self.headers, data=json_payload, timeout=None)
        response = await self.async_client.send(req, stream=True)

        color_print(f"Deepseek Web Response Status Code: {response.status_code}", "blue")
        # error will still return 200 status code, but only json format
        if response.headers.get("content-type") == "application/json":
            error_json = json.loads(await response.aread())
            raise HTTPException(status_code=400, detail=error_json)

        return response

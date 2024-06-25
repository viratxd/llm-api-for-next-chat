import base64
import hashlib
import json
import os
import random
import uuid
from datetime import datetime, timezone
from fastapi import HTTPException
from utility import color_print, get_user_agent
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.models import Response


class ChatGPT_Web_RE:
    openai_url = "https://chatgpt.com"
    proof_token = None
    async_session = AsyncSession()
    max_retries = 3
    alias_model = {"gpt-3.5": "text-davinci-002-render-sha"}

    def __init__(self):
        self.user_agent = get_user_agent()
        self.headers = {
            "User-Agent": self.user_agent,
            "Oai-Device-Id": str(uuid.uuid4()),
        }
        self.cookies = {
            "oai-did": self.headers["Oai-Device-Id"],
        }

    @property
    def is_anonymous(self) -> bool:
        return not bool(os.environ.get("CHATGPT_WEB_SESSION_TOKEN"))

    @property
    def backend_name(self) -> str:
        name = "anon" if self.is_anonymous else "api"
        return f"backend-{name}"

    async def _parse_openai_error_response(self, response: Response) -> dict:
        if response.astream_task:
            async for line in response.aiter_lines():
                if line:
                    error_response = json.loads(line.decode("utf-8"))
        else:
            error_response = response.json()
        if "detail" in error_response:
            return error_response["detail"]
        color_print(f"OpenAI Error Response: {error_response}", "red")
        return "Unknown error occurred."

    async def _set_access_token(self) -> None:
        self.cookies["__Secure-next-auth.session-token"] = os.environ["CHATGPT_WEB_SESSION_TOKEN"]
        auth_url = f"{self.openai_url}/api/auth/session"
        async with AsyncSession() as session:
            response = await session.get(auth_url, headers=self.headers, cookies=self.cookies, impersonate="chrome")
            response_json = response.json()
            if "accessToken" not in response_json:
                raise HTTPException(
                    status_code=400,
                    detail="Please provide a valid session token from https://chatgpt.com/api/auth/session cookie '__Secure-next-auth.session-token' value.",
                )
            self.headers["Authorization"] = f"Bearer {response_json['accessToken']}"

    def _generate_proof_token(self, seed: str, difficulty: str) -> str:
        prefix = "gAAAAAB"

        # generate config
        screen = random.randint(1000, 3000)
        now_utc = datetime.now(timezone.utc)
        parse_time = now_utc.strftime("%a, %d %b %Y %H:%M:%S GMT")
        config = [screen, parse_time, None, 0, self.user_agent]

        # generate answer
        diff_len = len(difficulty)
        for attempt in range(100000):
            config[3] = attempt
            json_str = json.dumps(config)
            answer = base64.b64encode(json_str.encode()).decode()
            candidate = hashlib.sha3_512((seed + answer).encode()).hexdigest()

            if candidate[:diff_len] <= difficulty:
                return prefix + answer

        # guarantee answer
        fallback_base = base64.b64encode(seed.encode()).decode()
        return prefix + "wQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D" + fallback_base

    async def _chat_requirements(self) -> dict:
        chat_requirements_url = f"{self.openai_url}/{self.backend_name}/sentinel/chat-requirements"

        async with AsyncSession() as session:
            response = await session.post(
                chat_requirements_url,
                json={"p": self.proof_token} if self.proof_token else {},
                headers=self.headers,
                impersonate="chrome",
            )
            if response.status_code == 401:
                # token expired, get new token
                color_print("Access token expired, getting new token...", "yellow")
                await self._set_access_token()
                return await self._chat_requirements()
            elif response.status_code != 200:
                color_print(f"Failed to get chat requirements: {response.status_code}", "red")
                raise HTTPException(
                    status_code=response.status_code, detail=await self._parse_openai_error_response(response)
                )
            response_json = response.json()
            if self.proof_token is None:
                pow = response_json["proofofwork"]
                self.proof_token = self._generate_proof_token(seed=pow["seed"], difficulty=pow["difficulty"])
                response_json = await self._chat_requirements()
            return response_json

    async def conversation(self, model: str, messages: str):
        if "Authorization" not in self.headers and not self.is_anonymous:
            await self._set_access_token()

        conversation_url = f"{self.openai_url}/{self.backend_name}/conversation"

        chat_requirements = await self._chat_requirements()
        self.headers["Openai-Sentinel-Chat-Requirements-Token"] = chat_requirements["token"]
        if chat_requirements.get("proofofwork") and chat_requirements["proofofwork"]["required"]:
            pow = chat_requirements["proofofwork"]
            self.headers["Openai-Sentinel-Proof-Token"] = self._generate_proof_token(
                seed=pow["seed"], difficulty=pow["difficulty"]
            )

        payload = {
            "action": "next",
            "model": self.alias_model.get(model, model),
            "parent_message_id": str(uuid.uuid4()),
            "messages": [
                {
                    "id": str(uuid.uuid4()),
                    "author": {"role": "user"},
                    "content": {
                        "content_type": "text",
                        "parts": [messages],
                    },
                }
            ],
            "conversation_id": None,
            "history_and_training_disabled": True,
            "conversation_mode": {
                "kind": "primary_assistant",
            },
            "force_paragen": False,
            "force_rate_limit": False,
            "websocket_request_id": str(uuid.uuid4()),
        }
        response = await self.async_session.request(
            "POST",
            conversation_url,
            json=payload,
            headers=self.headers,
            cookies=self.cookies,
            impersonate="chrome",
            stream=True,
            timeout=None,
        )
        color_print(f"ChatGPT Web Response Status Code: {response.status_code}", "blue")
        if response.status_code == 403 and self.max_retries > 0:
            color_print("Retrying chat request...", "yellow")
            self.max_retries -= 1
            self.proof_token = None
            response = await self.conversation(model, messages)
        elif response.status_code != 200 or self.max_retries == 0:
            raise HTTPException(
                status_code=response.status_code, detail=await self._parse_openai_error_response(response)
            )
        self.max_retries = 3
        return response

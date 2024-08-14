import base64
import hashlib
import json
import mimetypes
import os
import random
import re
import time
import uuid
from io import BytesIO
from datetime import datetime, timezone
from fastapi import HTTPException
from schemas import Message
from utility import color_print, get_user_agent
from curl_cffi import requests
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.models import Response
from PIL import Image


class ChatGPT_Web_RE:
    openai_url = "https://chatgpt.com"
    proof_token = None
    async_session = AsyncSession()
    max_retries = 3
    alias_model = {"gpt-3.5": "text-davinci-002-render-sha"}
    accepted_file_mime_types = []
    accepted_image_mime_types = []

    def __init__(self):
        self.user_agent = get_user_agent()
        self.headers = {"User-Agent": self.user_agent, "Oai-Device-Id": str(uuid.uuid4())}
        self.cookies = {"oai-did": self.headers["Oai-Device-Id"]}
        self._set_access_token()

    @property
    def is_anonymous(self) -> bool:
        return not bool(os.environ.get("CHATGPT_WEB_SESSION_TOKEN"))

    @property
    def backend_name(self) -> str:
        name = "anon" if self.is_anonymous else "api"
        return f"backend-{name}"

    async def _parse_openai_error_response(self, response: Response) -> dict | str:
        try:
            if response.astream_task:
                async for line in response.aiter_lines():
                    if line:
                        line_content = re.sub("^data: ", "", line.decode("utf-8"))
                        error_response = json.loads(line_content)
                        break
            else:
                error_response = response.json()
            if "detail" in error_response:
                color_print(f"OpenAI Error Response: {error_response}", "red")
                return error_response["detail"]
        except:
            pass
        return "Unknown error occurred."

    def _set_access_token(self) -> None:
        if not self.is_anonymous:
            self.cookies["__Secure-next-auth.session-token"] = os.environ["CHATGPT_WEB_SESSION_TOKEN"]
            auth_url = f"{self.openai_url}/api/auth/session"
            response = requests.get(auth_url, headers=self.headers, cookies=self.cookies, impersonate="chrome")
            response_json = response.json()
            if "accessToken" not in response_json:
                raise HTTPException(
                    status_code=400,
                    detail="Please provide a valid session token from https://chatgpt.com/api/auth/session cookie '__Secure-next-auth.session-token' value.",
                )
            self.headers["Authorization"] = f"Bearer {response_json['accessToken']}"

    async def _set_file_accept_type(self) -> None:
        # only for login users
        if not self.is_anonymous:
            async with AsyncSession() as session:
                response = await session.get(
                    f"{self.openai_url}/backend-api/models", headers=self.headers, impersonate="chrome"
                )
                if response.status_code != 200:
                    color_print(f"Failed to get models: {response.status_code}", "red")
                    raise HTTPException(
                        status_code=response.status_code, detail=await self._parse_openai_error_response(response)
                    )
                response_json = response.json()
                for model in response_json["models"]:
                    if model.get("product_features"):
                        attachments = model["product_features"]["attachments"]
                        self.accepted_file_mime_types = attachments["accepted_mime_types"]
                        self.accepted_image_mime_types = attachments["image_mime_types"]
                        break
                else:
                    color_print(f"Available models: {response_json}", "yellow")
                    raise HTTPException(
                        status_code=400,
                        detail="There are no available models with attachments now, maybe try again later.",
                    )

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
            if response.status_code == 401 and self.max_retries > 0:
                # token expired, get new token
                color_print("Access token expired, getting new token...", "yellow")
                self._set_access_token()
                self.max_retries -= 1
                return await self._chat_requirements()
            elif response.status_code != 200:
                color_print(f"Failed to get chat requirements: {response.status_code}", "red")
                self.max_retries = 3
                raise HTTPException(
                    status_code=response.status_code, detail=await self._parse_openai_error_response(response)
                )
            response_json = response.json()
            # reload until arkose is not required
            if self.proof_token is None or response_json.get("arkose"):
                color_print("Generating proof token...", "yellow")
                time.sleep(1)
                pow = response_json["proofofwork"]
                self.proof_token = self._generate_proof_token(seed=pow["seed"], difficulty=pow["difficulty"])
                response_json = await self._chat_requirements()
        return response_json

    async def _get_uploaded_file_detail(self, file_id: str) -> dict:
        file_detail_url = f"{self.openai_url}/backend-api/files/{file_id}"
        async with AsyncSession() as session:
            response = await session.get(file_detail_url, headers=self.headers, impersonate="chrome")
            if response.status_code != 200:
                raise Exception(f"Failed to get file details: {response.status_code}")
        return response.json()

    async def _upload_file(self, file_content, mime_type) -> dict:
        # generate file name
        file_extension = mimetypes.guess_extension(mime_type)
        random_file_name = uuid.uuid4().hex
        file_name = f"{random_file_name}{file_extension}"

        await self._set_file_accept_type()
        width = None
        height = None
        if mime_type in self.accepted_image_mime_types:
            try:
                with Image.open(BytesIO(file_content)) as img:
                    width, height = img.width, img.height
                file_use_cases = "multimodal"
            except Exception as e:
                color_print(f"Failed to get image dimensions: {e}, setting mime_type to text/plain", "yellow")
                mime_type = "text/plain"
        if mime_type in self.accepted_file_mime_types:
            file_use_cases = "my_files"
        elif mime_type not in self.accepted_image_mime_types + self.accepted_file_mime_types:
            file_use_cases = "ace_upload"
            mime_type = ""
            color_print(f"File type: {mime_type} not supported, setting mime_type to empty string", "yellow")

        async with AsyncSession() as session:
            # get url for file upload
            upload_api_url = f"{self.openai_url}/backend-api/files"
            file_size = len(file_content)
            upload_request_payload = {
                "file_name": file_name,
                "file_size": file_size,
                "use_case": file_use_cases,
            }
            upload_response = await session.post(
                upload_api_url, json=upload_request_payload, headers=self.headers, impersonate="chrome"
            )
            if upload_response.status_code != 200:
                raise HTTPException(
                    status_code=upload_response.status_code,
                    detail=await self._parse_openai_error_response(upload_response),
                )
            upload_data = upload_response.json()
            upload_url = upload_data.get("upload_url")
            file_id = upload_data.get("file_id")

            # upload file
            put_headers = {"Content-Type": mime_type, "X-Ms-Blob-Type": "BlockBlob"}  # Azure Blob Storage headers
            put_response = await session.put(upload_url, data=file_content, headers=put_headers, impersonate="chrome")
            if put_response.status_code != 201:
                raise HTTPException(
                    status_code=put_response.status_code, detail=await self._parse_openai_error_response(put_response)
                )

            # check file upload completion
            check_url = f"{self.openai_url}/backend-api/files/{file_id}/uploaded"
            check_response = await session.post(check_url, json={}, headers=self.headers, impersonate="chrome")
            if check_response.status_code != 200:
                raise HTTPException(
                    status_code=check_response.status_code,
                    detail=await self._parse_openai_error_response(check_response),
                )
            check_data = check_response.json()
            if check_data.get("status") != "success":
                raise HTTPException(status_code=400, detail="File upload completion check not successful")

            file_size_tokens = None
            # get fille_size_tokens
            if mime_type in self.accepted_file_mime_types:
                while True:
                    try:
                        file_detail = await self._get_uploaded_file_detail(file_id)
                        if file_detail.get("file_size_tokens"):
                            file_size = file_detail["size"]
                            file_size_tokens = file_detail["file_size_tokens"]
                            break
                        time.sleep(1)
                    except Exception:
                        raise HTTPException(status_code=400, detail="Failed to get file_size_tokens")

        return {
            "file_id": file_id,
            "file_name": file_name,
            "size_bytes": file_size,
            "mime_type": mime_type,
            "width": width,
            "height": height,
            "file_size_tokens": file_size_tokens,
        }

    async def _get_file_metadata(self, file_content, mime_type) -> dict:
        sha256_hash = hashlib.sha256(file_content).hexdigest()

        file_cache_path = "chatgpt_web/file_cache.json"
        # load file_cache.json file
        try:
            with open(file_cache_path, "r") as f:
                file_cache = json.load(f)
            cache_data = file_cache.get(sha256_hash)
            if cache_data:
                # check if file is still available in cloud
                try:
                    await self._get_uploaded_file_detail(cache_data["file_id"])
                    color_print(f"File found in cache: {cache_data}", "green")
                    return cache_data
                except Exception:
                    color_print("File has been deleted from cloud, re-uploading...", "yellow")
        except FileNotFoundError:
            file_cache = {}

        color_print("Uploading file...", "yellow")
        new_file_data = await self._upload_file(file_content, mime_type)
        color_print(f"File uploaded: {new_file_data}", "green")
        with open(file_cache_path, "w") as f:
            file_cache[sha256_hash] = new_file_data
            f.write(json.dumps(file_cache, indent=4))
        return new_file_data

    async def _generate_formatted_messages(self, messages: list[Message]) -> list:
        formatted_messages = []

        for message in messages:
            content = message.content
            new_parts = [content]
            content_type = "text"
            attachments = []

            if isinstance(content, list):
                new_parts = []

                for part in content:
                    if part.type == "text":
                        new_parts.append(part.text)
                    elif part.type == "image_url":
                        file_url = part.image_url.url
                        try:
                            # parse base64 data
                            if file_url.startswith("data:"):
                                mime_type, base64_data = (
                                    file_url.split(";")[0].split(":")[-1],
                                    file_url.split(",", 1)[-1],
                                )
                                file_content = base64.b64decode(base64_data)
                            # fetch image from URL
                            else:
                                tmp_headers = {"User-Agent": self.user_agent}
                                async with AsyncSession() as session:
                                    file_response = await session.get(
                                        file_url, headers=tmp_headers, impersonate="chrome"
                                    )
                                    file_content = file_response.content
                                    mime_type = file_response.headers.get("Content-Type", "").split(";")[0].strip()
                        except Exception as e:
                            color_print(f"Failed to fetch image: {e}", "red")
                            continue

                        file_metadata = await self._get_file_metadata(file_content, mime_type)
                        mime_type = file_metadata["mime_type"]
                        attachment = {
                            "name": file_metadata["file_name"],
                            "id": file_metadata["file_id"],
                            "mime_type": mime_type,
                            "size": file_metadata["size_bytes"],
                        }
                        content_type = "text"
                        if mime_type in self.accepted_image_mime_types:
                            content_type = "multimodal_text"
                            new_part = {
                                "asset_pointer": f"file-service://{file_metadata['file_id']}",
                                "content_type": "image_asset_pointer",
                                "height": file_metadata["height"],
                                "size_bytes": file_metadata["size_bytes"],
                                "width": file_metadata["width"],
                            }
                            new_parts.append(new_part)
                        elif mime_type in self.accepted_file_mime_types:
                            attachment["file_token_size"] = file_metadata["file_size_tokens"]

                        attachments.append(attachment)

            formatted_message = {
                "id": str(uuid.uuid4()),
                "author": {"role": message.role},
                "content": {"content_type": content_type, "parts": new_parts},
                "metadata": {"attachments": attachments} if attachments else {},
            }
            formatted_messages.append(formatted_message)

        return formatted_messages

    async def download_image(self, file_id: str) -> dict:
        url = f"{self.openai_url}/backend-api/files/{file_id}/download"
        async with AsyncSession() as session:
            response = await session.get(url, headers=self.headers, impersonate="chrome")
            download_details = response.json()
            image = await session.get(download_details["downlad_url"], headers=self.headers, impersonate="chrome")
        return {"file_name": download_details["file_name"], "image": image.content}

    async def conversation(self, model: str, messages: list[Message]) -> Response:
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
            "messages": await self._generate_formatted_messages(messages),
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
        if response.status_code == 403:
            if self.max_retries > 0:
                color_print("Retrying chat request...", "yellow")
                self.max_retries -= 1
                self.proof_token = None
                response = await self.conversation(model, messages)
            else:
                raise HTTPException(status_code=400, detail="Maximum retries reached. Please try again later.")
        elif response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, detail=await self._parse_openai_error_response(response)
            )
        self.max_retries = 3
        return response

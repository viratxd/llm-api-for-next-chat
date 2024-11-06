import json
import random
import string
import httpx
import os
from fastapi import HTTPException
from urllib3 import encode_multipart_formdata
from urllib3.fields import RequestField
from utility import color_print, get_user_agent


class HuggingChat_RE:
    hugging_face_url = "https://huggingface.co"
    chat_conversation_url = f"{hugging_face_url}/chat/conversation"
    model_key_mapping = {
        "llama-3.1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "command-r-plus-08-2024": "CohereForAI/c4ai-command-r-plus-08-2024",
        "qwen-2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
        "llama-3.1-nemotron-70b-instruct": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "llama-3.2-11b-vision-instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "hermes-3-llama-3.1-8b": "NousResearch/Hermes-3-Llama-3.1-8B",
        "mistralai-nemo-instruct-2407": "mistralai/Mistral-Nemo-Instruct-2407",
        "phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
    }
    web_search = False

    def __init__(self, async_client: httpx.AsyncClient = None) -> None:
        self.headers = {
            "Cookie": f"hf-chat={self.hf_chat}",
            "User-Agent": get_user_agent(),
            "Origin": self.hugging_face_url,
        }
        self.async_client = async_client or httpx.AsyncClient()
        self.conversation_id = None
        self.message_id = None

    @property
    def hf_chat(self) -> str:
        return os.environ.get("HUGGING_CHAT_TOKEN")

    @staticmethod
    def generate_random_boundary() -> str:
        return f"----WebKitFormBoundary{''.join(random.sample(string.ascii_letters + string.digits, 16))}"

    @property
    def config(self) -> list:
        config_id_mapping = {
            "Image Generation": "000000000000000000000001",
            "Document parser": "000000000000000000000002",
            "Image editor": "000000000000000000000003",
            "Calculator": "00000000000000000000000c",
            "Fetch URL": "00000000000000000000000b",
            "Web Search": "00000000000000000000000a",
        }

        config = []
        with open("hugging_chat/config.json", "r") as config_file:
            config_json = json.load(config_file)
            self.web_search = config_json["Web Search"]
            for tool_name, is_enabled in config_json.items():
                if is_enabled:
                    config.append(config_id_mapping[tool_name])
        return config

    async def _init_conversation(self, model: str, system_prompt: str) -> None:
        max_retries = 3
        retries = 0

        while retries <= max_retries:
            try:
                self.conversation_id = await self._find_conversation_id(model, system_prompt)
                self.message_id = await self._find_message_id()
                break
            except httpx.ReadTimeout:
                color_print("ReadTimeout Error: Retrying...", "yellow")
                retries += 1
        if retries == max_retries:
            color_print("Max retries exceeded. Unable to initialize conversation.", "red")
            raise HTTPException(status_code=500, detail="Unable to initialize conversation.")

    async def _find_conversation_id(self, model: str, system_prompt: str) -> str:
        payload = {"model": model, "preprompt": system_prompt}
        response = await self.async_client.post(self.chat_conversation_url, json=payload, headers=self.headers)
        if response.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid Hugging Face chat token.")
        response.raise_for_status()
        response_json = response.json()
        color_print(f"Initialised Conversation ID: {response_json['conversationId']}", "green")
        return response_json["conversationId"]

    async def _find_message_id(self) -> str:
        url = f"{self.chat_conversation_url}/{self.conversation_id}/__data.json?x-sveltekit-invalidated=11"
        response = await self.async_client.get(url, headers=self.headers)
        response.raise_for_status()
        response_json = json.loads(response.text.split("\n")[0])
        message_id = response_json["nodes"][1]["data"][3]
        color_print(f"Initialised Message ID: {message_id}", "green")
        return message_id

    async def delete_all_conversation(self) -> None:
        delete_url = f"{self.chat_conversation_url}s?/delete"
        headers = self.headers | {"Content-Type": f"multipart/form-data; boundary={self.generate_random_boundary()}"}
        response = await self.async_client.post(delete_url, headers=headers)
        response.raise_for_status()
        color_print("All conversation deleted.", "green")

    async def generate_image(self, sha: str):
        if not self.conversation_id:
            await self._init_conversation()
        url = f"{self.chat_conversation_url}/{self.conversation_id}/output/{sha}"
        response = await self.async_client.get(url, headers=self.headers)
        response.raise_for_status()
        return response

    async def request_conversation(
        self,
        query: str,
        model: str = "yi-1.5-34b-chat",
        system_prompt: str = "Act as an AI assistant that responds to user inputs in the language they use. Parse the provided JSON-formatted conversation history, but respond only to the final user message without referencing the JSON format. Maintain consistency with previous responses and adapt to the user's language preference.",
    ):
        if not self.hf_chat:
            raise HTTPException(status_code=400, detail="Please set the HUGGING_CHAT_TOKEN environment variable.")

        await self._init_conversation(
            model=self.model_key_mapping.get(model, "01-ai/Yi-1.5-34B-Chat"),
            system_prompt=system_prompt,
        )

        url = f"{self.chat_conversation_url}/{self.conversation_id}"

        request_fields = [
            RequestField(
                name="data",
                data=json.dumps(
                    {
                        "inputs": query,
                        "id": self.message_id,
                        "is_retry": False,
                        "is_continue": False,
                        "web_search": self.web_search,
                        "tools": self.config,
                    },
                    ensure_ascii=False,
                ),
                headers={"Content-Disposition": 'form-data; name="data"'},
            ),
        ]
        content, content_type = encode_multipart_formdata(request_fields, boundary=self.generate_random_boundary())
        headers = self.headers | {"Content-Type": content_type}
        req = self.async_client.build_request("POST", url, content=content, headers=headers, timeout=None)
        response = await self.async_client.send(req, stream=True)
        color_print(f"Hugging Chat Response Status Code: {response.status_code}", "blue")
        return response

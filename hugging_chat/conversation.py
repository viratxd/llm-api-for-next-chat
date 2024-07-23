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
        "command-r-plus": "CohereForAI/c4ai-command-r-plus",
        "llama-3-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
        "zephry-141b-a35b": "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
        "mixtral-8x7b-instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "nous-hermes-2-mixtral-8x7b-dpo": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "yi-1.5-34b-chat": "01-ai/Yi-1.5-34B-Chat",
        "gemma-1.1-7b-instruct": "google/gemma-1.1-7b-it",
        "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
        "phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    }

    def __init__(
        self,
        model: str = "command-r-plus",
        system_prompt: str = "Act as an AI assistant that responds to user inputs in the language they use. Parse the provided JSON-formatted conversation history, but respond only to the final user message without referencing the JSON format. Maintain consistency with previous responses and adapt to the user's language preference.",
        async_client: httpx.AsyncClient = None,
    ) -> None:
        self.model = self.model_key_mapping.get(model, "CohereForAI/c4ai-command-r-plus")
        self.system_prompt = system_prompt
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
    def config(self) -> dict:
        with open("hugging_chat/config.json", "r") as config_file:
            config = json.load(config_file)
        return config

    async def _init_conversation(self):
        max_retries = 3
        retries = 0

        while retries <= max_retries:
            try:
                self.conversation_id = await self._find_conversation_id()
                self.message_id = await self._find_message_id()
                break
            except httpx.ReadTimeout:
                color_print("ReadTimeout Error: Retrying...", "yellow")
                retries += 1
        if retries == max_retries:
            color_print("Max retries exceeded. Unable to initialize conversation.", "red")
            raise HTTPException(status_code=500, detail="Unable to initialize conversation.")

    async def _find_conversation_id(self) -> str:
        payload = {"model": self.model, "preprompt": self.system_prompt}
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
        response_json = response.json()
        message_id = response_json["nodes"][1]["data"][3]
        color_print(f"Initialised Message ID: {message_id}", "green")
        return message_id

    async def _delete_all_conversation(self) -> None:
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

    async def request_conversation(self, query: str):
        if not self.hf_chat:
            raise HTTPException(status_code=400, detail="Please set the HUGGING_CHAT_TOKEN environment variable.")

        await self._init_conversation()

        url = f"{self.chat_conversation_url}/{self.conversation_id}"

        config = self.config
        request_fields = [
            RequestField(
                name="data",
                data=json.dumps(
                    {
                        "inputs": query,
                        "id": self.message_id,
                        "is_retry": False,
                        "is_continue": False,
                        "web_search": config["websearch"],
                        "tools": config,
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
        await self._delete_all_conversation()
        return response

import json
import random
import string
import httpx
import os
from fastapi import HTTPException
from urllib3 import encode_multipart_formdata
from urllib3.fields import RequestField
from utility import get_user_agent


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


class HuggingChat_RE:
    hugging_face_url = "https://huggingface.co"
    chat_conversation_url = f"{hugging_face_url}/chat/conversation"

    def __init__(
        self, model: str = "command-r-plus", system_prompt: str = "", async_client: httpx.AsyncClient = None
    ) -> None:
        """
        Initializes an instance of the HuggingChat_RE class.

        Parameters:
        - hf_chat (str): The Hugging Face chat token.
        - model (str): The name or path of the model to be used for the chat. Defaults to "command-r-plus".
        - system_prompt (str): The system prompt to be used for the chat. Defaults to "Be Helpful and Friendly".
        - async_client (httpx.AsyncClient): An async client to be used for making HTTP requests. Defaults to None.

        Returns:
        - None: This is a constructor method and does not return anything.
        """
        self.hf_chat = os.environ.get("HUGGING_CHAT_ID")
        self.model = model_key_mapping.get(model, "CohereForAI/c4ai-command-r-plus")
        self.system_prompt = system_prompt
        self.headers = {
            "Cookie": f"hf-chat={self.hf_chat}",
            "User-Agent": get_user_agent(),
            "Origin": "https://huggingface.co",
        }
        self.async_client = async_client or httpx.AsyncClient()
        self.conversationId = None
        self.messageId = None

    async def init_conversation(self):
        max_retries = 3
        retries = 0

        while retries <= max_retries:
            try:
                self.conversationId = await self.find_conversation_id()
                self.messageId = await self.find_message_id()
                break
            except httpx.ReadTimeout:
                print("\033[91m" + "ReadTimeout Error: Retrying..." + "\033[0m")
                retries += 1
        if retries == max_retries:
            print("\033[91m" + "Max retries exceeded. Unable to initialize conversation." + "\033[0m")
            raise HTTPException(status_code=500, detail="Unable to initialize conversation.")

    async def find_conversation_id(self) -> str:
        """
        Finds and returns the conversation ID for the Hugging Face chat.

        Returns:
        - str: The conversation ID retrieved from the server response.
        """
        payload = {"model": self.model, "preprompt": self.system_prompt}
        response = await self.async_client.post(self.chat_conversation_url, json=payload, headers=self.headers)
        response.raise_for_status()
        response_json = response.json()
        print("\033[92m" + "Initialised Conversation ID:", response_json["conversationId"] + "\033[0m")
        return response_json["conversationId"]

    async def find_message_id(self) -> str:
        """
        Finds and returns the message ID for the Hugging Face chat.

        Returns:
        - str: The message ID retrieved from the server response.
        """
        url = f"{self.chat_conversation_url}/{self.conversationId}/__data.json?x-sveltekit-invalidated=11"
        response = await self.async_client.get(url, headers=self.headers)
        response.raise_for_status()
        response_json = response.json()
        print("\033[92m" + "Initialised Message ID:", response_json["nodes"][1]["data"][3] + "\033[0m")
        return response_json["nodes"][1]["data"][3]

    async def generate_image(self, sha: str):
        if not self.conversationId:
            await self.init_conversation()
        url = f"{self.chat_conversation_url}/{self.conversationId}/output/{sha}"
        response = await self.async_client.get(url, headers=self.headers)
        response.raise_for_status()
        return response

    async def request_conversation(self, query: str, web_search: bool = False):
        if not self.hf_chat:
            raise HTTPException(status_code=400, detail="Please set the HUGGING_CHAT_ID environment variable.")

        if not self.conversationId or not self.messageId:
            await self.init_conversation()

        url = f"{self.chat_conversation_url}/{self.conversationId}"

        request_fields = [
            RequestField(
                name="data",
                data=json.dumps(
                    {
                        "inputs": query,
                        "id": self.messageId,
                        "is_retry": False,
                        "is_continue": False,
                        "web_search": web_search,
                        "tools": {
                            "websearch": web_search,
                            "fetch_url": True,
                            "document_parser": True,
                            "query_calculator": False,
                            "image_editing": False,
                            "image_generation": True,
                        },
                    },
                    ensure_ascii=False,
                ),
                headers={"Content-Disposition": 'form-data; name="data"'},
            ),
        ]
        random_boundary = f"----WebKitFormBoundary{''.join(random.sample(string.ascii_letters + string.digits, 16))}"
        content, content_type = encode_multipart_formdata(request_fields, boundary=random_boundary)
        headers = self.headers | {"Content-Type": content_type}
        req = self.async_client.build_request("POST", url, content=content, headers=headers, timeout=None)
        response = await self.async_client.send(req, stream=True)
        print("Hugging Chat Response Status Code:", response.status_code)
        return response

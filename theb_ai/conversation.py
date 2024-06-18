import httpx
import json
import random
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from .Theb_AI_Login import Theb_API_JSON_PATH, start_async_tasks as generate_api_token
from schemas import MessageJsonData
from utility import get_user_agent


class TheB_AI_RE:
    theb_ai_api_url = "https://beta.theb.ai/api"
    model_key_mapping = {
        "TheB.AI": "theb-ai",
        "Claude 3 Opus": "claude-3-opus-20240229",
        "Claude 3 Sonnet": "claude-3-sonnet-20240229",
        "Claude 3 Haiku": "claude-3-haiku-20240307",
        "Llama 3 70B": "llama-3-70b",
        "Llama 3 8B": "llama-3-8b",
        "CodeLlama 70B": "codellama-70b",
        "CodeLlama 34B": "codellama-34b",
        "CodeLlama 13B": "codellama-13b",
        "CodeLlama 7B": "codellama-7b",
        "Mixtral 8x22B": "mixtral-8x22b",
        "Mixtral 8x7B": "mixtral-8x7b",
        "Mixtral 7B": "mixtral-7b",
        "WizardLM 2 8x22B": "wizardlm-2-8x22b",
        "DBRx Instruct": "dbrx-instruct",
        "Qwen1.5 110B": "qwen1.5-110b",
        "Qwen1.5 72B": "qwen1.5-72b",
        "Qwen1.5 32B": "qwen1.5-32b",
        "Qwen1.5 14B": "qwen1.5-14b",
        "Qwen1.5 7B": "qwen1.5-7b",
        "Yi 34B": "yi-34b",
    }

    def __init__(self, async_client: httpx.AsyncClient = None):
        api_info = self._get_api_info()
        self.organization_id, self.api_key = api_info["organization_id"], api_info["api_key"]
        self.headers = {"Authorization": f"Bearer {self.api_key}", "User-Agent": get_user_agent()}
        self.async_client = async_client or httpx.AsyncClient()

    async def _init_chat_models(self) -> dict[str, str]:
        chat_models = {}
        chat_models_url = f"{self.theb_ai_api_url}/chat_models"
        response = await self.async_client.get(chat_models_url, headers=self.headers)
        response.raise_for_status()
        response_json = response.json()
        for model in response_json["data"]:
            try:
                chat_models[self.model_key_mapping[model["model_name"]]] = model["model_id"]
            except KeyError:
                pass
        return chat_models

    def _load_api_info(self) -> list[dict]:
        try:
            with open(Theb_API_JSON_PATH, "r") as file:
                api_info = json.load(file)
            if len(api_info) == 0:
                raise Exception()
        except Exception:
            generate_api_token()
            return self._load_api_info()
        else:
            return api_info

    def _remove_apis(self) -> None:
        with open(Theb_API_JSON_PATH, "r") as file:
            data = json.load(file)
        if data:
            data.pop(0)
        with open(Theb_API_JSON_PATH, "w") as file:
            json.dump(data, file, indent=4)

    def _get_api_info(self) -> dict:
        api_info = self._load_api_info()
        return {"organization_id": api_info[0]["ORGANIZATION_ID"], "api_key": api_info[0]["API_KEY"]}

    async def conversation(self, message_json_data: MessageJsonData):
        chat_models = await self._init_chat_models()

        conversation_url = (
            f"{self.theb_ai_api_url}/conversation?org_id={self.organization_id}&req_rand={random.random()}"
        )
        request_payload = {
            "text": json.dumps(
                [jsonable_encoder(message) for message in message_json_data.messages], ensure_ascii=False
            ),
            # Default to Llama 3 8B
            "model": chat_models.get(message_json_data.model, "c60d009ce85f47f087952f17eead4eab"),
            "functions": [],
            "attachments": [],
            "model_params": {
                "system_prompt": "",
                "temperature": message_json_data.temperature,
                "top_p": message_json_data.top_p,
                "frequency_penalty": "0",
                "presence_penalty": "0",
                "long_term_memory": "ltm",
            },
        }
        json_payload = json.dumps(request_payload, ensure_ascii=False)
        req = self.async_client.build_request(
            "POST", conversation_url, headers=self.headers, data=json_payload, timeout=None
        )
        response = await self.async_client.send(req, stream=True)

        print("TheB AI Response Status Code:", response.status_code)
        if response.status_code == 400:
            # Sometimes TheB AI will update their models to require a minimum balance
            try:
                result_content = await response.aread()
                response_json = json.loads(result_content)
                if response_json["data"]["detail"].startswith("This model requires a minimum balance"):
                    raise Exception()
            except Exception:
                raise HTTPException(
                    status_code=400, detail="This model requires a minimum balance, please change the model."
                )
            else:
                print("API Token expired. Change and try again.")
                self._remove_apis()
                return await self.conversation(message_json_data)
        elif response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="API request failed")
        else:
            return response

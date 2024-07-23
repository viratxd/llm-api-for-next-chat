import httpx
import json
import random
from fastapi import HTTPException
from .Theb_AI_Login import Theb_API_JSON_PATH, start_async_tasks as generate_api_token
from utility import color_print, get_user_agent


class TheB_AI_RE:
    theb_ai_api_url = "https://beta.theb.ai/api"
    model_key_mapping = {
        "TheB.AI": "theb-ai",
        "Claude 3.5 Sonnet": "claude-3-5-sonnet-20240620",
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
    organization_id = None
    headers = None

    def __init__(self, async_client: httpx.AsyncClient = None):
        self.api_info = self._load_api_info()
        self._init_api_info()
        self.async_client = async_client or httpx.AsyncClient()

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

    def _init_api_info(self, index: int = 0) -> None:
        self.organization_id = self.api_info[index]["ORGANIZATION_ID"]
        self.headers = {
            "Authorization": f"Bearer {self.api_info[index]['API_KEY']}",
            "User-Agent": get_user_agent(),
        }

    def _remove_apis(self) -> None:
        with open(Theb_API_JSON_PATH, "r") as file:
            data = json.load(file)
        if data:
            data.pop(0)
        with open(Theb_API_JSON_PATH, "w") as file:
            json.dump(data, file, indent=4)

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

    async def _check_balance(self) -> float:
        url = f"{self.theb_ai_api_url}/organization/balance?org_id={self.organization_id}"
        response = await self.async_client.get(url, headers=self.headers)
        response.raise_for_status()
        response_json = response.json()
        balance = float(response_json["data"]["balance"])
        color_print(f"Current Balance: {balance}", "blue")
        if balance <= 0.0:
            color_print("Blance out of funds. Changing API token.", "yellow")
            self._remove_apis()
            self.api_info = self._load_api_info()
            self._init_api_info()
        return balance

    async def _select_target_balance_account(self, target_balance: float) -> None:
        for i in range(len(self.api_info)):
            self._init_api_info(i)
            balance = await self._check_balance()
            if balance >= target_balance:
                break
        # all accounts are not enough balance, generate new and choose the new one
        else:
            color_print("All accounts are not enough balance, generating new API token.", "yellow")
            generate_api_token()
            self.api_info = self._load_api_info()
            self.api_info.reverse()
            self._init_api_info()

    async def conversation(
        self, model: str = "llama-3-8b", text: str = "Hello!", temperature: float = 0.5, top_p: int = 1
    ):
        await self._check_balance()
        chat_models = await self._init_chat_models()

        conversation_url = (
            f"{self.theb_ai_api_url}/conversation?org_id={self.organization_id}&req_rand={random.random()}"
        )
        request_payload = {
            "text": text,
            # Default to Llama 3 8B
            "model": chat_models.get(model, "c60d009ce85f47f087952f17eead4eab"),
            "functions": [],
            "attachments": [],
            "model_params": {
                "system_prompt": "Act as an AI assistant that responds to user inputs in the language they use. Parse the provided JSON-formatted conversation history, but respond only to the final user message without referencing the JSON format. Maintain consistency with previous responses and adapt to the user's language preference.",
                "temperature": temperature,
                "top_p": top_p,
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

        color_print(f"TheB AI Response Status Code: {response.status_code}", "blue")
        if response.status_code != 200:
            # Sometimes TheB AI will update their models to require a minimum balance
            try:
                result_content = await response.aread()
                response_json = json.loads(result_content)
                detail = response_json["data"]["detail"]
                # search nubmer after $ sign
                required_min_balance = float(detail.split("$")[1])
                if required_min_balance > 0.05:
                    color_print(f"API error: {detail}", "yellow")
                    raise Exception("Required minimum balance is above trial limit.")
                await self._select_target_balance_account(required_min_balance)
                return await self.conversation(model, text, temperature, top_p)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        return response

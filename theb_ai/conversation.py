import httpx
import json
import random
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from .Theb_AI_Login import Theb_API_JSON_PATH, start_async_tasks as generate_api_token
from schemas import MessageJsonData
from utility import get_user_agent

# Mapping of model identifiers to their respective keys
model_key_mapping = {
    "theb-ai": "7e682da4dde7ee214baa0efc0cf6d7a4",  # free
    "claude-3-opus-20240229": "ade084104e614c31ada6892744fcd3c5",
    "claude-3-sonnet-20240229": "d2c2b37c042d4b248af62a033eccd1b2",
    "claude-3-haiku-20240307": "8b851ff081d54ba7b46a42a5099fcc64",
    "llama-3-70b": "5da08fc7ac704d0d9bee545cbbb91793",
    "llama-3-8b": "c60d009ce85f47f087952f17eead4eab",  # free
    "code-llama-70b": "bccdb1b4dee94dc59d6e15e2a73ac2ba",
    "code-llama-34b": "c4e13a6a0bc745bc92653147da2efea1",
    "code-llama-13b": "927f19df68ee4069b1ae53bc84a97ac5",
    "code-llama-7b": "63236596ff7243228ab7d6cdacbc990d",  # free
    "mixtral-8x22b": "70b3f32d71a34b97af9d660e1245fe15",
    "mixtral-8x7b": "5db5c052e7e44e339dc25f9fa2ef1224",
    "mixtral-7b": "61a9c92d0bf047fcbd3f87a5685c742f",  # free
    "wizardlm-2-8x22b": "ebf8820be60f40a38a3cae32795f8bd2",
    "dbrx-instruct": "20ea474d42dd44eaa618971cdb16bfa7",
    "qwen1.5-110b": "36bdcc531b5441b0ae79-34dee3311949",
    "qwen1.5-72b": "ac08d826a86a4723a4d04af0c1166d3a",
    "qwen1.5-32b": "63c3e577520d4fd9bb513e22cb528b1b",
    "qwen1.5-14b": "ff5aa63be5714586b45a21ee6134eb80",
    "qwen1.5-7b": "713bdc912fc54cf991fe2bf4fb3ea833",  # free
    "yi-34b": "1eabb6dc24fb4ed0a691f60954483eb8",
}


def load_api_info() -> list[dict]:
    try:
        with open(Theb_API_JSON_PATH, "r") as file:
            api_info = json.load(file)
        if len(api_info) == 0:
            raise Exception()
    except Exception:
        generate_api_token()
        return load_api_info()
    else:
        return api_info


def remove_apis() -> None:
    with open(Theb_API_JSON_PATH, "r") as file:
        data = json.load(file)
    if data:
        data.pop(0)
    with open(Theb_API_JSON_PATH, "w") as file:
        json.dump(data, file, indent=4)


def get_api_info() -> dict:
    api_info = load_api_info()
    return {"organization_id": api_info[0]["ORGANIZATION_ID"], "api_key": api_info[0]["API_KEY"]}


async def theb_ai_conversation(message_json_data: MessageJsonData, async_client: httpx.AsyncClient):
    api_info = get_api_info()
    organization_id, api_key = api_info["organization_id"], api_info["api_key"]
    api_endpoint = f"https://beta.theb.ai/api/conversation?org_id={organization_id}&req_rand={random.random()}"

    request_payload = {
        "text": json.dumps([jsonable_encoder(message) for message in message_json_data.messages], ensure_ascii=False),
        "model": model_key_mapping.get(message_json_data.model, "ade084104e614c31ada6892744fcd3c5"),
        "functions": None,
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
    req_headers = {"Authorization": f"Bearer {api_key}", "User-Agent": get_user_agent()}

    json_payload = json.dumps(request_payload, ensure_ascii=False)
    req = async_client.build_request("POST", api_endpoint, headers=req_headers, data=json_payload, timeout=None)
    response = await async_client.send(req, stream=True)
    print("TheB AI Response Status Code:", response.status_code)
    if response.status_code == 400:
        print("API Token expired. Change and try again.")
        remove_apis()
        return await theb_ai_conversation(message_json_data, async_client)
    elif response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="API request failed")
    else:
        return response

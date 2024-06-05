import base64
import uuid
from fastapi.encoders import jsonable_encoder
import httpx
import asyncio
import json
import re
import os
import time
from pydantic import ValidationError
from fastapi import FastAPI, BackgroundTasks, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv
from schemas import (
    MessageJsonData,
    CompletionsJsonData,
    TheB_Data,
    HuggingChatData,
    OpenAiData,
    Choices,
    Message,
    Usage,
    Content,
    ChunkJson,
)
from theb_ai.conversation import load_api_info, theb_ai_conversation, model_key_mapping as theb_ai_supported_models
from hugging_chat.conversation import HuggingChat_RE, model_key_mapping as hugging_chat_supported_models


class LoadEnvMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        load_dotenv(override=True)
        response = await call_next(request)
        return response


def get_response_headers(stream: bool):
    return {
        "Transfer-Encoding": "chunked",
        "X-Accel-Buffering": "no",
        "Content-Type": ("text/event-stream;" if stream else "application/json;" + " charset=utf-8"),
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_api_info()
    app.async_client = httpx.AsyncClient()
    yield
    await app.async_client.aclose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LoadEnvMiddleware)


@app.post("/api/anthropic/v1/messages")
async def anthropic_messages(message_json_data: MessageJsonData, background_tasks: BackgroundTasks):
    response = await theb_ai_conversation(message_json_data, app.async_client)
    background_tasks.add_task(response.aclose)
    lock = asyncio.Lock()

    async def content_generator():
        async with lock:
            accumulated_response_text = ""
            index = 0
            async for line in response.aiter_lines():
                if line and not line.startswith("event: "):
                    line_content = re.sub("^data: ", "", line)
                    try:
                        data = TheB_Data(**json.loads(line_content))
                    except json.JSONDecodeError:
                        continue
                    except ValidationError:
                        continue
                    text_delta = data.args.content.replace(accumulated_response_text, "")
                    yield (
                        f"data: {ChunkJson(delta=Content(type='text', text=text_delta), index=index).model_dump_json()}\n\n"
                        if message_json_data.stream
                        else text_delta
                    )
                    index += 1
                    accumulated_response_text = data.args.content
                if line == "event: end":
                    break

    response_headers = get_response_headers(message_json_data.stream)
    return (
        StreamingResponse(content_generator(), headers=response_headers)
        if message_json_data.stream
        # for summarize
        else JSONResponse(
            {"content": [{"text": "".join([text_delta async for text_delta in content_generator()])}]},
            headers=response_headers,
        )
    )


@app.post("/api/openai/v1/chat/completions")
async def openai_chat_completions(
    comletions_json_data: CompletionsJsonData, background_tasks: BackgroundTasks, authorization: str = Header("")
):
    async_client = app.async_client
    response_headers = get_response_headers(comletions_json_data.stream)

    # reverse to openai completions
    if comletions_json_data.model.startswith("gpt"):
        openai_api = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        req = async_client.build_request(
            "POST",
            openai_api,
            headers={"Authorization": authorization},
            data=comletions_json_data.model_dump_json(),
            timeout=None,
        )
        resp = await async_client.send(req, stream=True)
        background_tasks.add_task(resp.aclose)
        return StreamingResponse(resp.aiter_raw(), status_code=resp.status_code, headers=response_headers)
    # imitate to openai completions response
    else:
        # openai completions to claude message
        message_json_data = MessageJsonData(
            max_tokens=4000,
            messages=comletions_json_data.messages,
            model=comletions_json_data.model,
            stream=comletions_json_data.stream,
            temperature=comletions_json_data.temperature,
            top_k=5,
            top_p=comletions_json_data.top_p,
        )

        if message_json_data.model in theb_ai_supported_models:
            response = await theb_ai_conversation(message_json_data, async_client)
            lock = asyncio.Lock()

            async def content_generator():
                async with lock:
                    openai_data = OpenAiData(
                        choices=[Choices(delta=Message(role="assistant", content=""), index=0, finish_reason=None)],
                        created=int(time.time()),
                        id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
                        object="chat.completion.chunk",
                        model=message_json_data.model,
                    )
                    yield (
                        f"data: {openai_data.model_dump_json(exclude_unset=True)}\n\n"
                        if message_json_data.stream
                        else ""
                    )

                    accumulated_response_text = ""
                    async for line in response.aiter_lines():
                        if line and not line.startswith("event: "):
                            line_content = re.sub("^data: ", "", line)
                            try:
                                data = TheB_Data(**json.loads(line_content))
                            except (json.JSONDecodeError, ValidationError):
                                continue
                            text_delta = data.args.content.replace(accumulated_response_text, "")
                            openai_data.choices = [
                                Choices(delta=Message(content=text_delta), index=0, finish_reason=None)
                            ]
                            yield (
                                f"data: {openai_data.model_dump_json(exclude_unset=True)}\n\n"
                                if message_json_data.stream
                                else text_delta
                            )
                            accumulated_response_text = data.args.content
                        if line == "event: end":
                            openai_data.choices = [Choices(delta=Message(content=""), index=0, finish_reason="stop")]
                            yield (
                                f"data: {openai_data.model_dump_json(exclude_unset=True)}\n\ndata: [DONE]\n\n"
                                if message_json_data.stream
                                else ""
                            )
                            break

        elif message_json_data.model in hugging_chat_supported_models:
            hf_api = HuggingChat_RE(model=message_json_data.model, async_client=async_client)
            query = json.dumps(
                [jsonable_encoder(message) for message in message_json_data.messages], ensure_ascii=False
            )
            response = await hf_api.request_conversation(query)

            async def content_generator():
                openai_data = OpenAiData(
                    choices=[Choices(delta=Message(role="assistant", content=""), index=0, finish_reason=None)],
                    created=int(time.time()),
                    id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
                    object="chat.completion.chunk",
                    model=message_json_data.model,
                )
                yield (
                    f"data: {openai_data.model_dump_json(exclude_unset=True)}\n\n" if message_json_data.stream else ""
                )

                accumulated_response_text = ""
                async for line in response.aiter_lines():
                    data = HuggingChatData(**json.loads(line))
                    match data.type:
                        case "stream":
                            text_delta = data.token.replace(accumulated_response_text, "").replace("\u0000", "")
                            openai_data.choices = [
                                Choices(delta=Message(content=text_delta), index=0, finish_reason=None)
                            ]
                            yield (
                                f"data: {openai_data.model_dump_json(exclude_unset=True)}\n\n"
                                if message_json_data.stream
                                else text_delta
                            )
                            accumulated_response_text = data.token
                        case "file":
                            # convert image to base64 with markdown format
                            image = await hf_api.generate_image(data.sha)
                            image_base64 = base64.b64encode(image.content).decode("utf-8")
                            markdown_image = f"![{data.name}](data:{data.mime};base64,{image_base64})"
                            openai_data.choices = [
                                Choices(delta=Message(content=markdown_image), index=0, finish_reason=None)
                            ]
                            yield (
                                f"data: {openai_data.model_dump_json(exclude_unset=True)}\n\n"
                                if message_json_data.stream
                                else markdown_image
                            )
                        case "finalAnswer":
                            openai_data.choices = [Choices(delta=Message(content=""), index=0, finish_reason="stop")]
                            yield (
                                f"data: {openai_data.model_dump_json(exclude_unset=True)}\n\ndata: [DONE]\n\n"
                                if message_json_data.stream
                                else ""
                            )
                            break
                        case _:
                            print(f"Unparsed line: {line}")

        else:
            raise HTTPException(status_code=400, detail="Model not supported")

        background_tasks.add_task(response.aclose)
        return (
            StreamingResponse(content_generator(), headers=response_headers)
            if message_json_data.stream
            # for summarize
            else JSONResponse(
                OpenAiData(
                    choices=[
                        Choices(
                            message=Message(
                                role="assistant",
                                content="".join([text_delta async for text_delta in content_generator()]),
                            ),
                            index=0,
                            finish_reason="stop",
                        )
                    ],
                    created=int(time.time()),
                    id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
                    object="chat.completion",
                    model=message_json_data.model,
                    usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                ).model_dump(exclude_unset=True),
                headers=response_headers,
            )
        )

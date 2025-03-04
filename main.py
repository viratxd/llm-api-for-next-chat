import httpx
import asyncio
import json
import re
import time
import uuid
from pathlib import Path
from pydantic import ValidationError
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from environs import Env
from schemas import (
    CompletionsJsonData,
    OpenAiData,
    Choices,
    Message,
    Usage,
)
from deepseek_web.conversation import Deepseek_Web_RE
from utility import get_response_headers

env = Env()
env.read_env()

API_HOST = env("API_HOST", "http://localhost:5000")

async_client = httpx.AsyncClient()
deepseek_web = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Deepseek_Web_RE in a separate process
    global deepseek_web
    try:
        deepseek_web = await asyncio.to_thread(Deepseek_Web_RE.create)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Deepseek: {str(e)}")
    
    yield
    
    # Cleanup
    await async_client.aclose()
    if deepseek_web:
        await deepseek_web.async_session.aclose()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def load_env_middleware(request: Request, call_next):
    env.read_env(override=True)
    response = await call_next(request)
    return response

@app.get("/")
async def home():
    return {"message": "Welcome to the API Home Route"}

@app.post("/api/openai/v1/chat/completions")
async def openai_chat_completions(
    comletions_json_data: CompletionsJsonData,
    background_tasks: BackgroundTasks,
):
    model = comletions_json_data.model
    stream = comletions_json_data.stream
    response_headers = get_response_headers(stream)
    messages_str = json.dumps(
        [jsonable_encoder(message, exclude_unset=True) for message in comletions_json_data.messages],
        ensure_ascii=False,
    )

    if deepseek_web is None:
        raise HTTPException(status_code=500, detail="Server not initialized properly")

    if model in deepseek_web.models:
        response = await deepseek_web.completions(messages_str)

        async def content_generator():
            async for line in response.aiter_lines():
                if stream:
                    yield line.decode("utf-8") + "\n"
                else:
                    if line:
                        line_content = re.sub("^data: ", "", line)
                        try:
                            data = OpenAiData(**json.loads(line_content))
                        except (json.JSONDecodeError, ValidationError):
                            continue
                        yield data.choices[0].delta.content

    else:
        raise HTTPException(status_code=400, detail="Model not supported")

    background_tasks.add_task(response.aclose)
    return (
        StreamingResponse(content_generator(), headers=response_headers)
        if stream
        else JSONResponse(
            OpenAiData(
                choices=[
                    Choices(
                        message=Message(
                            role="assistant",
                            content="".join([text_delta async for text_delta in content_generator()]),
                        ),
                        finish_reason="stop",
                    )
                ],
                created=int(time.time()),
                id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
                object="chat.completion",
                model=model,
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            ).model_dump(exclude_unset=True),
            headers=response_headers,
        )
    )

@app.get("/image/{file_name}")
def image(file_name: str):
    return FileResponse(f"generated_images/{file_name}")

if __name__ == "__main__":
    import uvicorn
    port = int(env("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
import time
from typing import Literal
import uuid
from fake_useragent import UserAgent
from schemas import Choices, Message, OpenAiData


def color_print(text: str, color: Literal["red", "green", "yellow", "blue"]):
    colors = {"red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m", "blue": "\033[94m"}
    print(f"{colors[color]}{text}\033[0m")


def get_user_agent() -> str:
    ua = UserAgent()
    return ua.random


def get_response_headers(stream: bool):
    return {
        "Transfer-Encoding": "chunked",
        "X-Accel-Buffering": "no",
        "Content-Type": ("text/event-stream;" if stream else "application/json;" + " charset=utf-8"),
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }


def get_openai_chunk_response(model: str) -> OpenAiData:
    return OpenAiData(
        choices=[Choices(delta=Message(role="assistant", content=""))],
        created=int(time.time()),
        id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
        object="chat.completion.chunk",
        model=model,
    )


def get_openai_chunk_response_end(model: str, stream: bool) -> str:
    openai_data = get_openai_chunk_response(model)
    openai_data.choices = [Choices(delta=Message(content=""), finish_reason="stop")]
    return f"data: {openai_data.model_dump_json(exclude_unset=True)}\n\ndata: [DONE]\n\n" if stream else ""

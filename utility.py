from fake_useragent import UserAgent


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

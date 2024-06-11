from fake_useragent import UserAgent


def get_user_agent() -> str:
    ua = UserAgent()
    return ua.random

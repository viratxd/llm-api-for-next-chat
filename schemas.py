import base64
import io
from typing import Optional, Literal
from PIL import Image
from pydantic import BaseModel, field_validator


class Claude_ImageSource(BaseModel):
    data: str
    media_type: str
    type: str

    @field_validator("data")
    @classmethod
    def compress_image(cls, base64_image):
        # Convert base64 image to PIL Image
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes))

        # Resize image to max 512x512
        image.thumbnail((512, 512))

        # Convert PIL Image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()


class OpenAI_ImageURL(BaseModel):
    url: str


class Content(BaseModel):
    text: str = None
    source: Claude_ImageSource = None
    image_url: OpenAI_ImageURL = None
    type: str


class Author(BaseModel):
    role: str
    name: Optional[str] = None
    metadata: dict = {}


class ChatGPT_Web_Content(BaseModel):
    content_type: str = "text"
    parts: list[str]


class Message(BaseModel):
    role: str = None
    content: str | list[Content] | ChatGPT_Web_Content
    author: Optional[Author] = None
    status: Optional[str] = None


class MessageJsonData(BaseModel):
    max_tokens: int
    messages: list[Message]
    model: str
    stream: bool
    temperature: float
    top_k: int
    top_p: int


class CompletionsJsonData(BaseModel):
    frequency_penalty: float = 0
    messages: list[Message]
    model: str
    presence_penalty: float = 0
    stream: bool
    temperature: float
    top_p: int


class TheB_Data(BaseModel):
    id: str
    type: int
    args: Message


class ChunkJson(BaseModel):
    type: str = "content_block_delta"
    delta: Content
    index: int


class HuggingChatData(BaseModel):
    type: str = Literal["status", "title", "tool", "file", "stream", "finalAnswer"]
    token: Optional[str] = None
    name: Optional[str] = None
    sha: Optional[str] = None
    mime: Optional[str] = None


class Choices(BaseModel):
    delta: Message = None
    message: Message = None
    index: int = 0
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAiData(BaseModel):
    choices: list[Choices]
    created: int
    id: str
    object: str = Literal["chat.completion", "chat.completion.chunk"]
    model: str
    usage: Optional[Usage] = None

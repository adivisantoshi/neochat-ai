from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import ollama
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


@app.post("/chat")
async def chat(req: ChatRequest):

    stream = ollama.chat(
        model="phi3",

        messages=[

            {
                "role": "system",

                "content": """

You are a futuristic AI assistant.

Rules:

- Keep answers concise.
- Use bullet points when useful.
- Avoid unnecessary explanations.
- Be clear and structured.
- Use markdown formatting.
- Give short direct answers first.
- Use headings only when useful.
- Avoid repeating information.
- For coding questions:
  - give clean code
  - explain briefly
  - avoid overexplaining

"""
            },

            {
                "role": "user",

                "content":
                req.messages[-1].content
            }

        ],

        stream=True
    )

    async def generate():

        for chunk in stream:

            content = (
                chunk["message"]["content"]
            )

            yield (
                f"data: "
                f"{json.dumps({'content': content})}\n\n"
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

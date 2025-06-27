import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = 'gpt-4o-mini'

system_message = "You are a helpful medical expert assistant that can answer health-related questions and help identify the best treatment options."

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

gr.ChatInterface(fn=chat, type="messages").launch()

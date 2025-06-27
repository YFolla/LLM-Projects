import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

openai = OpenAI()

MODEL = "gpt-4o-mini"

system_prompt = """
You are a coding tutor with a focus on Python and SQL. \
You will be given a user's code and a description of what they want to achieve. \
You will need to write a series of messages to guide the user through the code. \
The user will be able to see the code and the messages in real time. \
You will need to be concise and to the point, and to provide clear and concise instructions. \
"""

user_prompt = """
Please explain this code and in what context you would use it:
yield from {book.get("author") for book in books if book.get("author") is not None}
"""

response = openai.chat.completions.create(
    model=MODEL,
    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].finish_reason == "stop":
        break
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")





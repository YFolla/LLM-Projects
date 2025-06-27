import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from openai import OpenAI

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

openai = OpenAI()

headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}


class Website:
    def __init__(self, url):
        self.url = url

        with sync_playwright() as p:
            browser = p.chromium.launch_persistent_context(
                user_data_dir="/tmp/playwright",
                headless=False,
                slow_mo=1000
            )
            page = browser.new_page()
            # Set extra HTTP headers
            page.set_extra_http_headers({
                "User-Agent": headers["User-Agent"],
                "Accept-Language": "en-US,en;q=0.9"
            })
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            self.title = page.title()
            html = page.content()
            browser.close()

        soup = BeautifulSoup(html, 'html.parser')
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

system_prompt = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."

def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "\nThe contents of this website is as follows; \
please provide a short summary of this website in markdown. \
If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt

def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]

def summarize(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = messages_for(website)
    )
    return response.choices[0].message.content

summary = summarize("https://openai.com")
print(summary)


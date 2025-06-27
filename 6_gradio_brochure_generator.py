import os
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
import anthropic
import gradio as gr

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

openai = OpenAI()
claude = anthropic.Anthropic()
google.generativeai.configure(api_key=google_api_key)

headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    def __init__(self, url):
        self.url = url

        with sync_playwright() as p:
            browser = p.chromium.launch_persistent_context(
                user_data_dir="/tmp/playwright",
                headless=True,
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
        
        # Extract links before removing irrelevant elements
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]
        
        # Clean up HTML content
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

def get_system_prompt(audience, tone):
    return f"You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for {audience} in a {tone} tone. Respond in markdown.\
Include details of company culture, customers and careers/jobs if you have the information."

def stream_gpt(prompt, audience, tone):
    system_message = get_system_prompt(audience, tone)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    stream = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result

def stream_claude(prompt, audience, tone):
    system_message = get_system_prompt(audience, tone)
    result = claude.messages.stream(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.7,
        system=system_message,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response = ""
    with result as stream:
        for text in stream.text_stream:
            response += text or ""
            yield response

def stream_gemini(prompt, audience, tone):
    system_message = get_system_prompt(audience, tone)
    model = google.generativeai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(
        f"{system_message}\n\n{prompt}",
        generation_config=google.generativeai.types.GenerationConfig(
            max_output_tokens=1000,
            temperature=0.7,
        )
    )
    yield response.text

def stream_brochure(company_name, url, model, audience, tone):
    prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
    prompt += Website(url).get_contents()
    if model=="GPT":
        result = stream_gpt(prompt, audience, tone)
    elif model=="Claude":
        result = stream_claude(prompt, audience, tone)
    elif model=="Gemini":
        result = stream_gemini(prompt, audience, tone)
    else:
        raise ValueError("Unknown model")
    yield from result

view = gr.Interface(
    fn=stream_brochure,
    inputs=[
        gr.Textbox(label="Company Name:", placeholder="Enter the company name"),
        gr.Textbox(label="Company Website URL:", placeholder="https://example.com"),
        gr.Dropdown(["GPT", "Claude", "Gemini"], label="Select model", value="GPT"),
        gr.Dropdown(["investors", "recruits", "customers"], label="Target audience", value="customers"),
        gr.Dropdown(["serious", "light-hearted", "business casual"], label="Tone", value="business casual")
    ],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never"
)
view.launch(share=True)


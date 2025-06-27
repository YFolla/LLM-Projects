import os
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from openai import OpenAI

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
MODEL = "gpt-4o-mini"

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
        
        # Extract links before removing irrelevant elements
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]
        
        # Clean up HTML content
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

# Extracting relevant links from the website

link_system_prompt = "You are provided with a list of links found on a webpage. \
You are able to decide which of the links would be most relevant to include in a brochure about the company, \
such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
link_system_prompt += "You should respond in JSON as in this example:"
link_system_prompt += """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""

def get_links_user_prompt(website):
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt

def get_links(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(website)}
      ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    return json.loads(result)

website = Website("https://n8n.io")
links = get_links(website.url)
print(links)

# Creating a brochure from the links and their contents

def convert_to_absolute_url(url, base_url):
    """Convert relative URLs to absolute URLs"""
    if url.startswith('http'):
        return url
    elif url.startswith('/'):
        # Remove trailing slash from base_url if present, then add the relative path
        return base_url.rstrip('/') + url
    else:
        # Handle other relative URLs (though less common)
        return base_url.rstrip('/') + '/' + url

def get_all_details(url):
    result = "Landing page:\n"
    website = Website(url)
    result += website.get_contents()
    
    links = get_links(url)
    print("Found links:", links)
    
    for link in links["links"]:
        try:
            # Convert relative URLs to absolute URLs
            absolute_url = convert_to_absolute_url(link["url"], url)
            print(f"Processing {link['type']}: {absolute_url}")
            
            result += f"\n\n{link['type']}\n"
            link_website = Website(absolute_url)
            result += link_website.get_contents()
        except Exception as e:
            print(f"Error processing {link['type']} ({link['url']}): {str(e)}")
            # Continue with other links even if one fails
            continue
    
    return result


system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
Include details of company culture, customers and careers/jobs if you have the information."

def get_brochure_user_prompt(company_name, url, all_details):
    user_prompt = f"You are looking at a company called: {company_name}\n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    user_prompt += all_details
    user_prompt = user_prompt[:15_000]  # Increased limit to capture more content
    return user_prompt

def create_brochure(company_name, url):
    all_details = get_all_details(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url, all_details)}
          ],
    )
    result = response.choices[0].message.content
    return result   

brochure = create_brochure('n8n', website.url)
print(brochure)
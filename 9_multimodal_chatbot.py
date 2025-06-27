import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from amadeus import Client
import base64
from io import BytesIO
from PIL import Image
from pydub import AudioSegment
from pydub.playback import play
import tempfile

load_dotenv()

MODEL = 'gpt-4o-mini'
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load Amadeus credentials
AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")

# Initialize Amadeus client
if not AMADEUS_API_KEY or not AMADEUS_API_SECRET:
    print("⚠️  WARNING: Amadeus API credentials not found in .env file!")
    print("Please ensure AMADEUS_API_KEY and AMADEUS_API_SECRET are set in your .env file")
    amadeus = None
else:
    try:
        amadeus = Client(
            client_id=AMADEUS_API_KEY,
            client_secret=AMADEUS_API_SECRET
        )
        print("✅ Amadeus SDK initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Amadeus SDK: {e}")
        amadeus = None

system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers. "
system_message += "Always be accurate. When users ask about flights, use the flight search tool to provide real-time information. "
system_message += "IMPORTANT: Whenever you search for flights to ANY destination, you MUST also call the generate_destination_image tool to create an image of that destination. This is required for every flight search. "
system_message += "When users show interest in a destination or ask about what a place looks like, use the image generation tool to create beautiful destination images."

# --- Tool Functions ---

def transcribe_audio(audio_file_path):
    """Transcribe audio to text using OpenAI's Whisper API."""
    if not audio_file_path or not os.path.exists(audio_file_path):
        return {"error": "Audio file not found or invalid path"}
    try:
        print(f"🎤 DEBUG: Transcribing audio file: {audio_file_path}")
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        transcribed_text = transcript.strip() if transcript else ""
        if not transcribed_text:
            return {"error": "No speech detected. Please speak clearly."}
        print(f"✅ DEBUG: Transcribed: '{transcribed_text}'")
        return {"success": True, "transcribed_text": transcribed_text}
    except Exception as e:
        print(f"❌ DEBUG: Transcription error: {e}")
        return {"error": f"Unable to transcribe audio: {e}"}

def generate_audio_response(message, voice="onyx"):
    """Generate audio speech from text using OpenAI's TTS API."""
    if not message or not isinstance(message, str) or len(message.strip()) == 0:
        return {"error": "Message text is required."}
    try:
        print(f"🔊 DEBUG: Generating audio for message: '{message[:50]}...'")
        response = openai.audio.speech.create(
            model="tts-1", voice=voice, input=message.strip()
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(response.content)
            print(f"✅ DEBUG: Audio saved to: {temp_file.name}")
            return {"success": True, "audio_file_path": temp_file.name}
    except Exception as e:
        print(f"❌ DEBUG: Audio generation error: {e}")
        return {"error": f"Unable to generate audio: {e}"}

def generate_destination_image(destination, style="vibrant pop-art"):
    """Generate an image representing a vacation destination."""
    if not destination or not isinstance(destination, str) or len(destination.strip()) == 0:
        return {"error": "Destination name is required."}
    try:
        prompt = f"An image representing a vacation in {destination}, showing tourist spots and everything unique about {destination}, in a {style} style"
        print(f"🎨 DEBUG: Generating image with prompt: '{prompt}'")
        image_response = openai.images.generate(
            model="dall-e-3", prompt=prompt, size="1024x1024", n=1, response_format="b64_json"
        )
        image_base64 = image_response.data[0].b64_json
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        print(f"✅ DEBUG: Image for {destination} generated successfully.")
        return {"success": True, "image": image}
    except Exception as e:
        print(f"❌ DEBUG: Image generation error: {e}")
        return {"error": f"Unable to generate image: {e}"}
        
def get_flight_offers(origin, destination, departure_date, adults, 
                     return_date=None, children=None, infants=None, 
                     travel_class=None, non_stop=None, currency_code=None, 
                     max_price=None, max_results=5):
    """Search for flight offers using the Amadeus SDK."""
    if not amadeus:
        return {"error": "Flight search service is currently unavailable."}
    try:
        search_params = {
            'originLocationCode': origin, 'destinationLocationCode': destination,
            'departureDate': departure_date, 'adults': adults, 'max': max_results
        }
        if return_date: search_params['returnDate'] = return_date
        # ... (add other optional params similarly for brevity)
        
        print(f"🔍 DEBUG: Calling Amadeus with params: {search_params}")
        response = amadeus.shopping.flight_offers_search.get(**search_params)
        print(f"✅ DEBUG: Found {len(response.data)} offers from Amadeus.")
        return {"success": True, "flights": response.data}
    except Exception as e:
        print(f"❌ DEBUG: Amadeus SDK error: {e}")
        return {"error": f"Unable to retrieve flight data: {e}"}

# --- Tool Definitions ---

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_flight_offers",
            "description": "Search for flight offers using real-time data",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Origin IATA code (e.g., 'JFK')"},
                    "destination": {"type": "string", "description": "Destination IATA code (e.g., 'LHR')"},
                    "departure_date": {"type": "string", "description": "Departure date in YYYY-MM-DD format"},
                    "adults": {"type": "integer", "description": "Number of adult passengers"},
                    "return_date": {"type": "string", "description": "Return date for round trips"},
                },
                "required": ["origin", "destination", "departure_date", "adults"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_destination_image",
            "description": "Generate a beautiful image of a vacation destination",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "The city or destination name"},
                    "style": {"type": "string", "description": "Artistic style (e.g., 'vibrant pop-art')"},
                },
                "required": ["destination"],
            },
        },
    },
]
available_tools = {
    "get_flight_offers": get_flight_offers,
    "generate_destination_image": generate_destination_image,
}

# --- Core Chat Logic ---

def chat_with_tools(message, history):
    """Main chat function that handles user messages and tool calls."""
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    response_message = response.choices[0].message

    # Process tool calls if any
    if response_message.tool_calls:
        messages.append(response_message)
        generated_image = None

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_tools[function_name]
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"🤖 DEBUG: Calling tool '{function_name}' with args: {function_args}")
            function_response = function_to_call(**function_args)
            
            # Store image if generated, but pass only serializable data to the model
            if function_name == "generate_destination_image" and function_response.get("success"):
                generated_image = function_response.pop("image", None)

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response),
            })
        
        second_response = openai.chat.completions.create(model=MODEL, messages=messages)
        final_text = second_response.choices[0].message.content
        return final_text, generated_image

    # If no tool call, return the direct response
    return response_message.content, None

# --- Gradio Interface ---

with gr.Blocks(title="FlightAI - Multimodal Travel Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛫 FlightAI - Your Multimodal Travel Assistant")
    gr.Markdown("Search flights, generate destination images, and interact with voice! 🎤🔊")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History", height=500, type="messages")
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type a message or use the microphone...",
                    label="Your Message",
                    lines=3,
                    scale=4,
                )
                audio_input = gr.Audio(
                    sources=["microphone"], type="filepath", label="🎤", scale=1
                )
            send_btn = gr.Button("Send Message", variant="primary")
        
        with gr.Column(scale=1):
            image_output = gr.Image(label="Destination Image", height=300)
            audio_output = gr.Audio(label="Assistant Voice", autoplay=True)
            with gr.Row():
                voice_selector = gr.Dropdown(
                    choices=["onyx", "alloy", "echo", "fable", "nova", "shimmer"],
                    value="onyx", label="Voice", scale=2
                )
                audio_enabled = gr.Checkbox(value=True, label="Enable Audio", scale=1)
            status_display = gr.Textbox(label="Status", value="Ready!", interactive=False, lines=2)

    def respond(message, audio_file, history, voice_choice, audio_enabled):
        """Unified function to handle both text and audio inputs."""
        user_message = ""
        
        if audio_file:
            status_display.value = "Transcribing your voice..."
            transcription_result = transcribe_audio(audio_file)
            if transcription_result.get("success"):
                user_message = transcription_result["transcribed_text"]
                history.append({"role": "user", "content": f"🎤: {user_message}"})
            else:
                error = transcription_result.get("error", "Transcription failed.")
                return history, "", None, None, f"Error: {error}"
        elif message.strip():
            user_message = message
            history.append({"role": "user", "content": user_message})
        else:
            return history, "", None, None, "Ready! Ask a question."

        status_display.value = "Thinking..."
        response_text, generated_image = chat_with_tools(user_message, history)
        history.append({"role": "assistant", "content": response_text})

        audio_file_path = None
        if audio_enabled:
            status_display.value = "Generating audio response..."
            audio_result = generate_audio_response(response_text, voice_choice)
            if audio_result.get("success"):
                audio_file_path = audio_result["audio_file_path"]
        
        return history, "", None, generated_image, audio_file_path, "Response generated."

    send_btn.click(
        respond,
        inputs=[msg, audio_input, chatbot, voice_selector, audio_enabled],
        outputs=[chatbot, msg, audio_input, image_output, audio_output, status_display]
    )
    
    msg.submit(
        respond,
        inputs=[msg, audio_input, chatbot, voice_selector, audio_enabled],
        outputs=[chatbot, msg, audio_input, image_output, audio_output, status_display]
    )

demo.launch()


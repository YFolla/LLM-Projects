# Required imports
import os
import gradio as gr
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
from google.colab import drive, userdata
from huggingface_hub import login

# Constants
AUDIO_MODEL = "whisper-1"
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Initialize OpenAI client
openai_client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

# Login to Hugging Face
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

# Quantization config for the LLaMA model
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

def process_audio_file(audio_file, progress=gr.Progress()):
    """
    Process an uploaded audio file and generate meeting minutes.
    """
    if audio_file is None:
        return "❌ Please upload an audio file."
    
    try:
        progress(0.1, desc="Starting transcription...")
        
        # Transcribe audio using Whisper (using your existing openai client)
        with open(audio_file, "rb") as audio:
            transcription = openai.audio.transcriptions.create(
                model=AUDIO_MODEL,
                file=audio,
                response_format="text"
            )
        
        progress(0.5, desc="Transcription completed. Generating summary...")
        
        # Generate meeting minutes using LLaMA
        system_message = """You are an assistant that produces professional meeting minutes from transcripts. 
        Create well-structured minutes in markdown format including:
        - Meeting summary with attendees, location, and date (if mentioned)
        - Key discussion points
        - Important decisions made
        - Action items with owners (if mentioned)
        - Next steps"""
        
        user_prompt = f"""Below is a transcript of a meeting. Please write professional meeting minutes in markdown format:

{transcription}"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
        
        progress(0.7, desc="Generating meeting minutes...")
        
        # Use your existing model setup
        tokenizer = AutoTokenizer.from_pretrained(LLAMA)
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True
        ).to("cuda")
        
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA, 
            device_map="auto", 
            quantization_config=quant_config
        )
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=2000,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode and extract assistant response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_text = tokenizer.decode(inputs[0], skip_special_tokens=True)
        
        if input_text in full_response:
            summary = full_response.replace(input_text, "").strip()
        else:
            summary = full_response.split("assistant")[-1].strip() if "assistant" in full_response else full_response
        
        progress(1.0, desc="Meeting minutes generated successfully!")
        return summary
        
    except Exception as e:
        return f"❌ An error occurred: {str(e)}"

def process_audio_from_drive(audio_drive_path, progress=gr.Progress()):
    """
    Process an audio file from Google Drive path.
    """
    if not audio_drive_path or not audio_drive_path.strip():
        return "❌ Please enter the path to your audio file."
    
    # Clean the path
    audio_drive_path = audio_drive_path.strip()
    if audio_drive_path.startswith('/'):
        audio_drive_path = audio_drive_path[1:]
    
    full_audio_path = f"/content/drive/MyDrive/{audio_drive_path}"
    
    if not os.path.exists(full_audio_path):
        return f"❌ Error: The file '{audio_drive_path}' was not found in your Google Drive. Please ensure the path is correct and the file exists."
    
    try:
        progress(0.1, desc="Reading file from Google Drive...")
        
        with open(full_audio_path, "rb") as audio_file:
            transcription = openai.audio.transcriptions.create(
                model=AUDIO_MODEL,
                file=audio_file,
                response_format="text"
            )
        
        progress(0.5, desc="Transcription completed. Generating summary...")
        
        system_message = """You are an assistant that produces professional meeting minutes from transcripts. 
        Create well-structured minutes in markdown format including:
        - Meeting summary with attendees, location, and date (if mentioned)
        - Key discussion points
        - Important decisions made
        - Action items with owners (if mentioned)
        - Next steps"""
        
        user_prompt = f"""Below is a transcript of a meeting. Please write professional meeting minutes in markdown format:

{transcription}"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
        
        progress(0.7, desc="Generating meeting minutes...")
        
        # Use your existing model setup
        tokenizer = AutoTokenizer.from_pretrained(LLAMA)
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True
        ).to("cuda")
        
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA, 
            device_map="auto", 
            quantization_config=quant_config
        )
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=2000,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_text = tokenizer.decode(inputs[0], skip_special_tokens=True)
        
        if input_text in full_response:
            summary = full_response.replace(input_text, "").strip()
        else:
            summary = full_response.split("assistant")[-1].strip() if "assistant" in full_response else full_response
        
        progress(1.0, desc="Meeting minutes generated successfully!")
        return summary
        
    except Exception as e:
        return f"❌ An error occurred: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Meeting Minutes Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 Meeting Minutes Generator")
    gr.Markdown("Generate professional meeting minutes from audio files using Whisper and LLaMA models.")
    
    # Main interface with tabs
    with gr.Tabs():
        # Tab 1: File Upload
        with gr.TabItem("📤 Upload Audio File"):
            gr.Markdown("### Upload an audio file directly")
            with gr.Row():
                with gr.Column():
                    audio_upload = gr.Audio(
                        label="Upload Audio File",
                        type="filepath"
                    )
                    upload_btn = gr.Button("Generate Minutes", variant="primary")
                    
                with gr.Column():
                    upload_output = gr.Markdown(
                        label="Meeting Minutes",
                        value="Upload an audio file and click 'Generate Minutes' to see the results here."
                    )
        
        # Tab 2: Google Drive Path
        with gr.TabItem("🔗 Google Drive Path"):
            gr.Markdown("### Enter path to audio file in Google Drive")
            gr.Markdown("**Example paths:**")
            gr.Markdown("- `LLM_files/meeting.mp3`")
            gr.Markdown("- `recordings/team_call.wav`")
            gr.Markdown("- `audio/presentation.m4a`")
            
            with gr.Row():
                with gr.Column():
                    drive_path = gr.Textbox(
                        label="Google Drive File Path",
                        placeholder="e.g., LLM_files/my_meeting.mp3",
                        lines=1
                    )
                    drive_btn = gr.Button("Generate Minutes", variant="primary")
                    
                with gr.Column():
                    drive_output = gr.Markdown(
                        label="Meeting Minutes",
                        value="Enter a file path and click 'Generate Minutes' to see the results here."
                    )
    
    # Button click handlers
    upload_btn.click(
        process_audio_file,
        inputs=[audio_upload],
        outputs=[upload_output]
    )
    
    drive_btn.click(
        process_audio_from_drive,
        inputs=[drive_path],
        outputs=[drive_output]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True, debug=True)
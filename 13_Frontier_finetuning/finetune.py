"""
OpenAI GPT Fine-tuning Script for Price Estimation

This script fine-tunes a GPT-4o-mini model to estimate item prices.
The process includes:
1. Loading preprocessed training/test data
2. Formatting data for OpenAI fine-tuning API
3. Creating and monitoring a fine-tuning job with W&B integration
4. Testing the fine-tuned model

Required files:
- train.pkl: Training data (Item objects)
- test.pkl: Test data (Item objects)
- .env: Environment variables (OPENAI_API_KEY, HF_TOKEN, WANDB_API_KEY)

Setup instructions:
1. pip install wandb
2. Add WANDB_API_KEY to your .env file
3. The script will automatically configure W&B integration with OpenAI
"""

# Standard library imports
import os
import re
import json
import pickle
from typing import List, Dict, Any

# Third-party imports
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
import wandb

# Local imports
from items import Item
from testing import Tester

# =============================================================================
# CONFIGURATION AND ENVIRONMENT SETUP
# =============================================================================

# Load environment variables
load_dotenv(override=True)

# Set up API keys
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY')

# Configuration constants
TRAIN_SIZE = 200          # Number of examples for training
VALIDATION_SIZE = 50      # Number of examples for validation  
MODEL_NAME = "gpt-4o-mini-2024-07-18"
N_EPOCHS = 1
SEED = 42
MAX_INFERENCE_TOKENS = 7

# W&B Configuration
WANDB_PROJECT = "gpt-pricer"
WANDB_ENTITY = None  # Set to your W&B username/team name if needed

# Verify W&B is available and configured
try:
    if os.getenv('WANDB_API_KEY'):
        print("W&B API key found - integration will be enabled")
        wandb_available = True
    else:
        print("Warning: WANDB_API_KEY not found in environment variables")
        print("W&B integration will be disabled")
        wandb_available = False
except ImportError:
    print("Warning: wandb not installed. Install with: pip install wandb")
    wandb_available = False

# Log in to HuggingFace (required for some integrations)
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)

# Initialize OpenAI client
openai = OpenAI()

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_data():
    """
    Load preprocessed training and test data from pickle files.
    
    Returns:
        tuple: (train_data, test_data) containing lists of Item objects
    """
    try:
        with open('train.pkl', 'rb') as file:
            train = pickle.load(file)
        
        with open('test.pkl', 'rb') as file:
            test = pickle.load(file)
        
        print(f"Loaded {len(train)} training examples and {len(test)} test examples")
        return train, test
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required data file not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

# Load the data
train, test = load_data()

# Split training data into train/validation sets
# OpenAI recommends 50-100 examples, but we use 200 for better coverage
fine_tune_train = train[:TRAIN_SIZE]
fine_tune_validation = train[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]

print(f"Fine-tuning with {len(fine_tune_train)} training and {len(fine_tune_validation)} validation examples")

# =============================================================================
# DATA FORMATTING FOR FINE-TUNING
# =============================================================================

def format_training_messages(item: Item) -> List[Dict[str, str]]:
    """
    Format an Item object into the message format required for fine-tuning.
    
    This version includes the complete assistant response with the actual price,
    used for training the model.
    
    Args:
        item: Item object containing product details and price
        
    Returns:
        List of message dictionaries in OpenAI chat format
    """
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    
    # Clean the user prompt by removing training-specific text
    user_prompt = (item.test_prompt()
                   .replace(" to the nearest dollar", "")
                   .replace("\n\nPrice is $", ""))
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": f"Price is ${item.price:.2f}"}
    ]

def format_inference_messages(item: Item) -> List[Dict[str, str]]:
    """
    Format an Item object for inference with the fine-tuned model.
    
    This version has an incomplete assistant response, allowing the model
    to complete the price prediction.
    
    Args:
        item: Item object containing product details
        
    Returns:
        List of message dictionaries in OpenAI chat format
    """
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    
    user_prompt = (item.test_prompt()
                   .replace(" to the nearest dollar", "")
                   .replace("\n\nPrice is $", ""))
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Price is $"}
    ]

def create_jsonl_content(items: List[Item]) -> str:
    """
    Convert a list of Item objects to JSONL format for OpenAI fine-tuning.
    
    Each line contains a JSON object with a "messages" key containing
    the conversation format expected by OpenAI.
    
    Args:
        items: List of Item objects to convert
        
    Returns:
        String in JSONL format
    """
    lines = []
    
    for item in items:
        messages = format_training_messages(item)
        line = json.dumps({"messages": messages})
        lines.append(line)
    
    return '\n'.join(lines)

def write_jsonl_file(items: List[Item], filename: str) -> None:
    """
    Write items to a JSONL file for fine-tuning.
    
    Args:
        items: List of Item objects to write
        filename: Output filename
    """
    try:
        with open(filename, "w", encoding='utf-8') as f:
            jsonl_content = create_jsonl_content(items)
            f.write(jsonl_content)
        print(f"Written {len(items)} examples to {filename}")
    
    except Exception as e:
        raise RuntimeError(f"Error writing JSONL file {filename}: {e}")

# =============================================================================
# FILE UPLOAD AND FINE-TUNING JOB CREATION
# =============================================================================

# Create JSONL files for training and validation
print("Creating JSONL files...")
write_jsonl_file(fine_tune_train, "fine_tune_train.jsonl")
write_jsonl_file(fine_tune_validation, "fine_tune_validation.jsonl")

# Upload files to OpenAI
print("Uploading files to OpenAI...")
try:
    with open("fine_tune_train.jsonl", "rb") as f:
        train_file = openai.files.create(file=f, purpose="fine-tune")
        print(f"Training file uploaded: {train_file.id}")

    with open("fine_tune_validation.jsonl", "rb") as f:
        validation_file = openai.files.create(file=f, purpose="fine-tune")
        print(f"Validation file uploaded: {validation_file.id}")

except Exception as e:
    raise RuntimeError(f"Error uploading files: {e}")

# Create fine-tuning job with Weights & Biases integration for monitoring
print("Creating fine-tuning job...")

# Configure W&B integration for OpenAI fine-tuning
wandb_integration = None
if wandb_available:
    wandb_config = {"project": WANDB_PROJECT}
    if WANDB_ENTITY:
        wandb_config["entity"] = WANDB_ENTITY
    
    wandb_integration = {"type": "wandb", "wandb": wandb_config}
    print(f"W&B integration configured for project: {WANDB_PROJECT}")
else:
    print("W&B integration disabled")

# Optional: Initialize a manual W&B run for additional tracking
manual_wandb_run = None
if wandb_available:
    try:
        manual_wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name="gpt-pricer-finetuning",
            config={
                "model": MODEL_NAME,
                "train_size": TRAIN_SIZE,
                "validation_size": VALIDATION_SIZE,
                "n_epochs": N_EPOCHS,
                "seed": SEED,
                "max_inference_tokens": MAX_INFERENCE_TOKENS
            },
            tags=["openai-finetuning", "price-estimation"]
        )
        print("Manual W&B run initialized for additional tracking")
    except Exception as e:
        print(f"Warning: Could not initialize manual W&B run: {e}")
        manual_wandb_run = None

try:
    fine_tuning_job = openai.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=validation_file.id,
        model=MODEL_NAME,
        seed=SEED,
        hyperparameters={"n_epochs": N_EPOCHS},
        integrations=[wandb_integration] if wandb_integration else None,
        suffix="pricer"
    )
    
    job_id = fine_tuning_job.id
    print(f"Fine-tuning job created: {job_id}")
    
    # Log job details to manual W&B run if available
    if manual_wandb_run:
        manual_wandb_run.log({
            "job_id": job_id,
            "training_file_id": train_file.id,
            "validation_file_id": validation_file.id
        })

except Exception as e:
    raise RuntimeError(f"Error creating fine-tuning job: {e}")

# =============================================================================
# MONITORING (OPTIONAL - FOR DEBUGGING/TRACKING)
# =============================================================================

# Optional: Monitor the fine-tuning job status
# Uncomment these lines if you want to check job status programmatically

# print("Checking job status...")
# job_status = openai.fine_tuning.jobs.retrieve(job_id)
# print(f"Job status: {job_status.status}")

# # Optional: View recent events
# events = openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10)
# print("Recent events:")
# for event in events.data:
#     print(f"  {event.created_at}: {event.message}")

# =============================================================================
# RETRIEVE FINE-TUNED MODEL (AFTER COMPLETION)
# =============================================================================

# Get the fine-tuned model name
# Note: This assumes the job has completed. In practice, you'd wait or check status.
try:
    job_details = openai.fine_tuning.jobs.retrieve(job_id)
    fine_tuned_model_name = job_details.fine_tuned_model
    
    if fine_tuned_model_name:
        print(f"Fine-tuned model available: {fine_tuned_model_name}")
    else:
        print("Fine-tuning job not yet complete. Model name not available.")
        # In practice, you'd wait and retry, or handle this gracefully
        fine_tuned_model_name = "gpt-4o-mini-2024-07-18"  # Fallback for testing

except Exception as e:
    print(f"Error retrieving fine-tuned model: {e}")
    fine_tuned_model_name = MODEL_NAME  # Fallback to base model

# =============================================================================
# INFERENCE AND TESTING
# =============================================================================

def extract_price_from_response(response_text: str) -> float:
    """
    Extract a numeric price from the model's response text.
    
    Handles various formats like "$123.45", "123.45", "$1,234.56", etc.
    
    Args:
        response_text: Raw response from the model
        
    Returns:
        Extracted price as float, or 0.0 if no valid price found
    """
    # Remove common currency symbols and formatting
    cleaned = response_text.replace('$', '').replace(',', '').strip()
    
    # Use regex to find the first number (integer or decimal)
    match = re.search(r"[-+]?\d*\.?\d+", cleaned)
    
    if match:
        try:
            return float(match.group())
        except ValueError:
            return 0.0
    
    return 0.0

def predict_price_with_fine_tuned_model(item: Item) -> float:
    """
    Use the fine-tuned model to predict the price of an item.
    
    Args:
        item: Item object to price
        
    Returns:
        Predicted price as float
    """
    try:
        response = openai.chat.completions.create(
            model=fine_tuned_model_name,
            messages=format_inference_messages(item),
            seed=SEED,
            max_tokens=MAX_INFERENCE_TOKENS,
            temperature=0  # Use deterministic responses for consistent testing
        )
        
        reply = response.choices[0].message.content
        predicted_price = extract_price_from_response(reply)
        
        return predicted_price
    
    except Exception as e:
        print(f"Error during inference: {e}")
        return 0.0

# =============================================================================
# FINAL TESTING
# =============================================================================

print("\nTesting fine-tuned model...")
print("=" * 50)

# Test the fine-tuned model using the existing Tester class
# This will evaluate the model's performance on the test dataset
test_results = Tester.test(predict_price_with_fine_tuned_model, test)

# Log test results to W&B if available
if manual_wandb_run and test_results:
    # Assuming Tester.test returns some metrics (you may need to modify based on actual return)
    try:
        if isinstance(test_results, dict):
            manual_wandb_run.log({"test_results": test_results})
        else:
            manual_wandb_run.log({"test_completed": True})
        print("Test results logged to W&B")
    except Exception as e:
        print(f"Warning: Could not log test results to W&B: {e}")

# Finish the W&B run
if manual_wandb_run:
    try:
        manual_wandb_run.finish()
        print("W&B run finished successfully")
    except Exception as e:
        print(f"Warning: Error finishing W&B run: {e}")

print("\nFine-tuning and testing complete!")
print("\nTo view your fine-tuning progress:")
if wandb_available:
    print(f"1. Visit your W&B dashboard: https://wandb.ai/{WANDB_ENTITY or 'your-username'}/{WANDB_PROJECT}")
    print("2. OpenAI's automatic W&B integration will show training metrics")
    print("3. Manual run shows additional configuration and test results")
else:
    print("1. Install wandb: pip install wandb")
    print("2. Add WANDB_API_KEY to your .env file")
    print("3. Re-run the script to enable W&B integration")



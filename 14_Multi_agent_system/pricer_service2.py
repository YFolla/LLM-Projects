"""
Modal app for a fine-tuned pricing service using Llama 3.1 8B model.

This module provides a cloud-based pricing service that uses a fine-tuned Llama model
to extract price information from product descriptions. The service uses PEFT (Parameter
Efficient Fine-Tuning) to load a specialized pricing model on top of the base Llama model.
"""

import modal
from modal import App, Volume, Image

# ============================================================================
# MODAL INFRASTRUCTURE CONFIGURATION
# ============================================================================

# Create Modal app instance for the pricing service
app = modal.App("pricer-service")

# Define container image with all required dependencies
# - huggingface: For model hub access
# - torch: PyTorch for deep learning operations
# - transformers: Hugging Face library for model loading and tokenization
# - bitsandbytes: For 4-bit quantization to reduce memory usage
# - accelerate: For efficient model loading and device mapping
# - peft: Parameter Efficient Fine-Tuning library for loading adapters
image = Image.debian_slim().pip_install(
    "huggingface", 
    "torch", 
    "transformers", 
    "bitsandbytes", 
    "accelerate", 
    "peft"
)

# Secrets configuration - requires HuggingFace token for model access
# Note: Create this secret in Modal dashboard with your HF token
# Depending on your Modal configuration, you may need to replace "hf-secret" with "huggingface-secret"
secrets = [modal.Secret.from_name("hf-secret")]

# ============================================================================
# MODEL AND HARDWARE CONFIGURATION
# ============================================================================

# Hardware configuration
GPU = "T4"  # NVIDIA T4 GPU - good balance of performance and cost

# Base model configuration
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"  # Foundation model

# Fine-tuned model configuration
PROJECT_NAME = "pricer"  # Project identifier for the fine-tuned model
HF_USER = "YFolla"  # HuggingFace username - replace with your username or use this for reproduction
RUN_NAME = "2025-07-02_21.56.00"  # Timestamp identifier for the specific training run
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"  # Combined project and run identifier
REVISION = None  # Git revision/branch to use (None for latest)
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"  # Full path to fine-tuned model

# Storage configuration
CACHE_DIR = "/cache"  # Directory for caching models and data

# Scaling configuration
# Set to 1 if you want Modal to keep containers warm, otherwise containers go cold after 2 minutes
MIN_CONTAINERS = 0

# ============================================================================
# PROMPT CONFIGURATION
# ============================================================================

# Standard question prompt for price extraction
QUESTION = "How much does this cost to the nearest dollar?"

# Expected prefix in model response for price parsing
PREFIX = "Price is $"

# ============================================================================
# PERSISTENT STORAGE SETUP
# ============================================================================

# Create persistent volume for HuggingFace model cache
# This prevents re-downloading models on every container startup
hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

# ============================================================================
# PRICING SERVICE CLASS
# ============================================================================

@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),  # Set HF cache directory
    secrets=secrets,                                # HuggingFace authentication
    gpu=GPU,                                       # GPU type for acceleration
    timeout=1800,                                  # 30-minute timeout for long operations
    min_containers=MIN_CONTAINERS,                 # Minimum containers to keep warm
    volumes={CACHE_DIR: hf_cache_volume}          # Mount persistent cache volume
)
class Pricer:
    """
    A pricing service class that uses a fine-tuned Llama model to extract prices from descriptions.
    
    This class implements a Modal service that:
    1. Loads a base Llama model with 4-bit quantization for memory efficiency
    2. Applies a fine-tuned PEFT adapter specialized for pricing tasks
    3. Provides a price extraction method that processes product descriptions
    
    The service is designed to be stateful, keeping the model loaded in memory
    between requests for better performance.
    """

    @modal.enter()
    def setup(self):
        """
        Initialize the pricing model during container startup.
        
        This method runs once when the container starts and loads:
        1. The base Llama model with 4-bit quantization
        2. The tokenizer with proper padding configuration
        3. The fine-tuned PEFT adapter for pricing tasks
        
        The @modal.enter() decorator ensures this runs during container initialization,
        not on every method call, improving performance for subsequent requests.
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from peft import PeftModel
        
        # ====================================================================
        # QUANTIZATION CONFIGURATION
        # ====================================================================
        
        # Configure 4-bit quantization for memory efficiency
        # This reduces model size from ~16GB to ~4GB with minimal quality loss
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,       # Use double quantization for better accuracy
            bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computations
            bnb_4bit_quant_type="nf4"            # Use NF4 quantization type (recommended)
        )

        # ====================================================================
        # BASE MODEL AND TOKENIZER LOADING
        # ====================================================================
        
        # Load tokenizer for text preprocessing
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        # Configure tokenizer padding settings
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS token for padding
        self.tokenizer.padding_side = "right"                # Pad sequences on the right side
        
        # Load base model with quantization and automatic device mapping
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            quantization_config=quant_config,  # Apply 4-bit quantization
            device_map="auto"                  # Automatically distribute across available devices
        )
        
        # ====================================================================
        # FINE-TUNED MODEL LOADING
        # ====================================================================
        
        # Load the fine-tuned PEFT adapter on top of the base model
        # This adds task-specific parameters without modifying the base model
        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model,    # Base model to adapt
            FINETUNED_MODEL,    # Path to fine-tuned adapter
            revision=REVISION   # Specific revision/branch (None for latest)
        )

    @modal.method()
    def price(self, description: str) -> float:
        """
        Extract price from a product description using the fine-tuned model.
        
        This method processes a product description through the fine-tuned model
        and extracts a numerical price value from the generated response.
        
        Args:
            description (str): Product description text to analyze for pricing
            
        Returns:
            float: Extracted price value in dollars, or 0 if no price found
            
        Process:
            1. Format the description with a standard pricing prompt
            2. Generate response using the fine-tuned model
            3. Parse the response to extract numerical price
            4. Return the price as a float value
        """
        import os
        import re
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from peft import PeftModel
    
        # ====================================================================
        # TEXT GENERATION
        # ====================================================================
        
        # Set random seed for reproducible generation
        set_seed(42)
        
        # Format the prompt with question, description, and expected prefix
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
        
        # Tokenize input prompt and move to GPU
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        
        # Create attention mask (all tokens are attended to)
        attention_mask = torch.ones(inputs.shape, device="cuda")
        
        # Generate price prediction using the fine-tuned model
        outputs = self.fine_tuned_model.generate(
            inputs,                           # Input token IDs
            attention_mask=attention_mask,    # Attention mask
            max_new_tokens=5,                # Generate only 5 new tokens (sufficient for price)
            num_return_sequences=1           # Return single sequence
        )
        
        # Decode generated tokens back to text
        result = self.tokenizer.decode(outputs[0])
    
        # ====================================================================
        # PRICE EXTRACTION AND PARSING
        # ====================================================================
        
        # Extract content after the expected price prefix
        contents = result.split("Price is $")[1]
        
        # Remove commas from price string (e.g., "1,000" -> "1000")
        contents = contents.replace(',', '')
        
        # Use regex to find the first number (integer or decimal) in the response
        # Pattern matches: optional sign, optional digits, decimal point, digits OR just digits
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        
        # Return the extracted price as float, or 0 if no valid number found
        return float(match.group()) if match else 0
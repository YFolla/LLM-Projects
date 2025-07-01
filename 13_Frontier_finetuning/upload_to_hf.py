# Upload Curated Dataset to HuggingFace Hub
# This script loads the locally saved pickle files and uploads them to HuggingFace

import os
import pickle
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import Dataset, DatasetDict

# ===== ENVIRONMENT CONFIGURATION =====
load_dotenv(override=True)
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# ===== HUGGINGFACE AUTHENTICATION =====
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)

print("🔄 Loading locally saved datasets...")

# ===== LOAD PICKLE FILES =====
with open('train.pkl', 'rb') as file:
    train = pickle.load(file)
print(f"✅ Loaded {len(train):,} training items")

with open('test.pkl', 'rb') as file:
    test = pickle.load(file)
print(f"✅ Loaded {len(test):,} test items")

# ===== PREPARE DATA FOR HUGGINGFACE =====
print("🔄 Preparing data for HuggingFace format...")

# Extract prompts and prices
train_prompts = [item.prompt for item in train]
train_prices = [item.price for item in train]
test_prompts = [item.test_prompt() for item in test]
test_prices = [item.price for item in test]

# Create HuggingFace datasets
train_dataset = Dataset.from_dict({"text": train_prompts, "price": train_prices})
test_dataset = Dataset.from_dict({"text": test_prompts, "price": test_prices})
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

print("✅ HuggingFace datasets created")

# ===== PUSH TO HUGGINGFACE HUB =====
HF_USER = "YFolla"
DATASET_NAME = f"{HF_USER}/pricer-data"

print(f"🚀 Uploading dataset to {DATASET_NAME}...")

try:
    dataset.push_to_hub(DATASET_NAME, private=True)
    print(f"🎉 Successfully uploaded dataset to HuggingFace Hub!")
    print(f"📍 Dataset available at: https://huggingface.co/datasets/{DATASET_NAME}")
except Exception as e:
    print(f"❌ Error uploading to HuggingFace Hub: {e}")
    exit(1)

print("\n✅ Upload complete!") 
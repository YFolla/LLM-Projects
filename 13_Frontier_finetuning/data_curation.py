# Data Curation Pipeline for LLM Fine-tuning
# This script processes Amazon product data to create a balanced training dataset
# for fine-tuning an LLM to predict product prices

# ===== IMPORTS AND SETUP =====

import os
import random
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
from collections import Counter, defaultdict
import pickle

# ===== ENVIRONMENT CONFIGURATION =====

# Load environment variables for API keys
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# ===== HUGGINGFACE AUTHENTICATION =====

# Log in to HuggingFace to access datasets
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)

# ===== IMPORT CUSTOM MODULES =====
# Import after HF login to ensure authentication is complete

from loaders import ItemLoader
from items import Item

# ===== MAIN EXECUTION =====
# Wrap main code in __main__ block to fix multiprocessing issues

if __name__ == '__main__':
    # ===== MAIN DATA LOADING PIPELINE =====
    # Load and process multiple product categories using ItemLoader

    # Define the product categories to include in the training dataset
    dataset_names = [
        "Automotive",
        "Electronics",
        "Office_Products",
        "Tools_and_Home_Improvement",
        "Cell_Phones_and_Accessories",
        "Toys_and_Games",
        "Appliances",
        "Musical_Instruments",
    ]

    # Load all datasets in parallel and combine into single list
    items = []
    for dataset_name in dataset_names:
        loader = ItemLoader(dataset_name)
        items.extend(loader.load(workers=8))  # Optimized for 10-core system

    print(f"A grand total of {len(items):,} items")

    # ===== DATA BALANCING AND SAMPLING =====
    # Create balanced dataset by organizing items by price and applying sampling strategy

    # Group items by rounded price ($1, $2, $3, etc.) for balanced sampling
    slots = defaultdict(list)
    for item in items:
        slots[round(item.price)].append(item)

    # ===== BALANCED SAMPLING STRATEGY =====
    # Create a more evenly distributed dataset across price ranges
    # Apply different sampling rules based on price and category

    # Set random seed for reproducible results
    np.random.seed(42)
    random.seed(42)

    sample = []
    for i in range(1, 1000):  # For each price point from $1 to $999
        slot = slots[i]
        
        if i >= 240:  # For expensive items ($240+), include all available
            sample.extend(slot)
        elif len(slot) <= 1200:  # For price points with few items, include all
            sample.extend(slot)
        else:  # For price points with many items, sample strategically
            # Give lower weight to Automotive items (they dominate the dataset)
            # Give higher weight to other categories for better balance
            weights = np.array([1 if item.category=='Automotive' else 5 for item in slot])
            weights = weights / np.sum(weights)
            
            # Sample 1200 items from this price point using weighted selection
            selected_indices = np.random.choice(len(slot), size=1200, replace=False, p=weights)
            selected = [slot[i] for i in selected_indices]
            sample.extend(selected)

    print(f"There are {len(sample):,} items in the sample")

    # ===== TRAIN/TEST SPLIT =====
    # Randomly shuffle and split the balanced sample into training and test sets

    random.seed(42)
    random.shuffle(sample)

    # Create train/test split: 400k for training, 2k for testing
    train = sample[:400_000]
    test = sample[400_000:402_000]
    print(f"Divided into a training set of {len(train):,} items and test set of {len(test):,} items")

    # ===== PREPARE TRAINING DATA =====
    # Extract prompts and prices for training and testing

    # Training data: full prompts with answers
    train_prompts = [item.prompt for item in train]
    train_prices = [item.price for item in train]

    # Test data: prompts without answers (for evaluation)
    test_prompts = [item.test_prompt() for item in test]
    test_prices = [item.price for item in test]

    # ===== DATASET QUALITY VALIDATION =====
    # Perform quality checks before saving to ensure dataset integrity

    print("\n===== DATASET QUALITY VALIDATION =====")

    # 1. Check category distribution
    print("1. Category Distribution:")
    category_counts = Counter(item.category for item in train)
    for category, count in category_counts.most_common():
        percentage = (count / len(train)) * 100
        print(f"   {category}: {count:,} items ({percentage:.1f}%)")

    # 2. Check price distribution
    print("\n2. Price Distribution:")
    train_prices_array = np.array(train_prices)
    print(f"   Min price: ${train_prices_array.min():.2f}")
    print(f"   Max price: ${train_prices_array.max():.2f}")
    print(f"   Mean price: ${train_prices_array.mean():.2f}")
    print(f"   Median price: ${np.median(train_prices_array):.2f}")

    # 3. Check price range coverage
    print("\n3. Price Range Coverage:")
    price_ranges = [
        (1, 10, "$1-$10"),
        (11, 50, "$11-$50"), 
        (51, 100, "$51-$100"),
        (101, 250, "$101-$250"),
        (251, 500, "$251-$500"),
        (501, 1000, "$501+")
    ]

    for min_price, max_price, label in price_ranges:
        count = sum(1 for price in train_prices if min_price <= price <= max_price)
        percentage = (count / len(train_prices)) * 100
        print(f"   {label}: {count:,} items ({percentage:.1f}%)")

    # 4. Check prompt format consistency
    print("\n4. Prompt Format Validation:")
    sample_prompts = train_prompts[:100]  # Check first 100 prompts
    format_issues = 0

    for i, prompt in enumerate(sample_prompts):
        if not prompt.startswith(Item.QUESTION):
            format_issues += 1
        if not prompt.endswith(".00"):
            format_issues += 1

    print(f"   Format issues found: {format_issues}/200 checks")

    # 5. Check token count distribution
    print("\n5. Token Count Distribution:")
    token_counts = [item.token_count for item in train[:1000]]  # Sample 1000 items
    token_counts_array = np.array(token_counts)
    print(f"   Min tokens: {token_counts_array.min()}")
    print(f"   Max tokens: {token_counts_array.max()}")
    print(f"   Mean tokens: {token_counts_array.mean():.1f}")

    # 6. Sample prompts for manual inspection
    print("\n6. Sample Training Prompts:")
    for i in range(3):
        print(f"\n--- Sample {i+1} ---")
        print(f"Category: {train[i].category}")
        print(f"Price: ${train[i].price}")
        print(f"Tokens: {train[i].token_count}")
        print(f"Prompt preview: {train[i].prompt[:200]}...")

    print("\n7. Test Set Validation:")
    print(f"   Test prompts end with '{Item.PREFIX}': {all(prompt.endswith(Item.PREFIX) for prompt in test_prompts[:10])}")
    print(f"   Test set size: {len(test):,} items")

    print("\n===== VALIDATION COMPLETE =====")
    validation_passed = (
        len(train) > 0 and len(test) > 0 and
        format_issues < 10 and  # Allow some minor format issues
        len(category_counts) >= 5  # Should have multiple categories
    )

    if validation_passed:
        print("✅ Dataset quality validation PASSED - Ready to save!")
    else:
        print("❌ Dataset quality validation FAILED - Review issues above")
        exit(1)

    # ===== CREATE HUGGINGFACE DATASETS =====
    # Convert lists to HuggingFace Dataset format for easy use with transformers

    train_dataset = Dataset.from_dict({"text": train_prompts, "price": train_prices})
    test_dataset = Dataset.from_dict({"text": test_prompts, "price": test_prices})
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # ===== SAVE DATASETS =====
    # Option 1: Push to HuggingFace Hub (currently commented out)
    HF_USER = "YFolla"
    DATASET_NAME = f"{HF_USER}/pricer-data"
    dataset.push_to_hub(DATASET_NAME, private=True)

    # Option 2: Save locally as pickle files for immediate use
    print("\n===== SAVING DATASETS =====")
    with open('train.pkl', 'wb') as file:
        pickle.dump(train, file)
    print("✅ Training set saved to train.pkl")

    with open('test.pkl', 'wb') as file:
        pickle.dump(test, file)
    print("✅ Test set saved to test.pkl")

    print(f"\n🎉 Dataset curation complete! {len(train):,} training items and {len(test):,} test items saved locally.")
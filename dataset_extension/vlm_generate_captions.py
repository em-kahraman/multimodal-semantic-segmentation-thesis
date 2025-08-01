import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import os
import json
from tqdm import tqdm

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'datasets', 'NYUDepthv2'))

TRAINING_FILE = os.path.join(DATASET_DIR, 'test.txt')
OUTPUT_FILE = os.path.join(DATASET_DIR, 'generated_rgb_descriptions.json')

PROMPT = "USER: <image>\nWrite a strictly factual description of the image, focus on spatial relations and object positions.\nASSISTANT:"

# Model Loading (with 4-bit quantization)
print("Loading model...")
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    device_map={"": 0}
)

# Optional: use the line below to verify the device map
print(model.hf_device_map)

processor = AutoProcessor.from_pretrained(model_id)
print("Model loaded successfully.")

# Data Processing Loop

# Read the list of files from train.txt
try:
    with open(TRAINING_FILE, 'r') as f:
        # We only care about the first item on each line (the RGB path)
        rgb_files_relative = [line.strip().split()[0] for line in f if line.strip()]
    print(f"Found {len(rgb_files_relative)} RGB images to process in {TRAINING_FILE}")
except FileNotFoundError:
    print(f"Error: The training file was not found at {TRAINING_FILE}")
    exit()

# Load existing data if the output file already exists to allow resuming
all_descriptions = []
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        all_descriptions = json.load(f)
    print(f"Resumed from existing file with {len(all_descriptions)} entries.")

# Create a set of already processed RGB paths for quick lookup
processed_rgb_paths = {entry['rgb_path'] for entry in all_descriptions}

try:
    # Iterate over the relative RGB paths from the text file
    for relative_rgb_path in tqdm(rgb_files_relative, desc="Generating Descriptions"):
        
        # Construct full path by joining the dataset directory with the relative path
        full_rgb_path = os.path.join(DATASET_DIR, relative_rgb_path)

        # Skip files that are already processed
        if full_rgb_path in processed_rgb_paths:
            continue

        # Check that the RGB file actually exists before proceeding
        if not os.path.exists(full_rgb_path):
            print(f"Warning: Skipping, RGB image not found at: {full_rgb_path}")
            continue
            
        try:
            # Prepare model inputs
            image = Image.open(full_rgb_path).convert("RGB")
            inputs = processor(text=PROMPT, images=image, return_tensors="pt")
            inputs = inputs.to(model.device)

            # Generate description
            output = model.generate(**inputs, max_new_tokens=300)
            generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]
            
            # Clean up the output to get only the assistant's response
            description = generated_text.split("ASSISTANT:")[1].strip()

            # Store the simplified data (only RGB path and description)
            all_descriptions.append({
                "rgb_path": full_rgb_path,
                "description": description
            })
            processed_rgb_paths.add(full_rgb_path)

        except Exception as e:
            print(f"Error processing {full_rgb_path}: {e}")

except KeyboardInterrupt:
    print("\nProcess interrupted by user. Saving progress...")

finally:
    # Save all collected descriptions to the JSON file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_descriptions, f, indent=4)
    print(f"\nProcessing complete. {len(all_descriptions)} descriptions saved to {OUTPUT_FILE}")

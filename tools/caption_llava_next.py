import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import os
import json
from tqdm import tqdm
import argparse
import logging
from itertools import islice

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate image captions using LLaVA-Next.")
    
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf",
                        help="The Hugging Face model ID for LLaVA-Next.")
    parser.add_argument("--dataset_dir", type=str, default="datasets/NYUDepthv2",
                        help="Path to the root of the dataset directory.")
    parser.add_argument("--input_files", nargs='+', default=['train.txt', 'test.txt'],
                        help="One or more input files. Defaults to ['train.txt', 'test.txt'].")
    parser.add_argument("--output_file", type=str, default="captions/NYUDepthv2/llava-next.json",
                        help="Path of the output JSON file to save descriptions.")
    parser.add_argument("--prompt", type=str, 
                        default="USER: <image>\nWrite a strictly factual description of the image, focusing on object positions and spatial relations.\nASSISTANT:",
                        help="The prompt to use for the VLM.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of images to process in a single batch.")
    parser.add_argument("--max_new_tokens", type=int, default=300,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--quantize", action='store_true',
                        help="Enable 4-bit quantization to save VRAM.")
    
    return parser.parse_args()

def load_model_and_processor(model_id, quantize=False):
    """Load LLaVA-NeXT and its processor, optionally quantised to 4 bit."""
    logging.info(f"Loading model: {model_id} (Quantization: {quantize})")
    
    quantization_config = None
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)
    logging.info("Model and processor loaded successfully.")
    return model, processor

def batched(iterable, n):
    """Yield tuples of n items from an iterable."""
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            return
        yield chunk

def main():
    args = parse_arguments()
    model, processor = load_model_and_processor(args.model_id, args.quantize)
    output_file_path = args.output_file
    os.makedirs(os.path.dirname(output_file_path) or ".", exist_ok=True)

    # Image paths of the requested splits (train.txt, test.txt)
    all_relative_paths = []
    for file_name in args.input_files:
        try:
            file_path = os.path.join(args.dataset_dir, file_name)
            with open(file_path, 'r') as f:
                paths = [line.strip().split()[0] for line in f if line.strip()]
                all_relative_paths.extend(paths)
                logging.info(f"Read {len(paths)} image paths from {file_name}")
        except FileNotFoundError:
            logging.error(f"Error: The input file was not found at {file_path}")
    
    all_relative_paths = sorted(list(set(all_relative_paths)))
    logging.info(f"Found a total of {len(all_relative_paths)} unique image paths to process.")

    # Resume from an existing output file, skipping images that already have a caption
    all_descriptions = []
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as f:
            all_descriptions = json.load(f)
        logging.info(f"Resumed from existing file with {len(all_descriptions)} entries.")

    processed_rgb_paths = {entry['rgb_path'] for entry in all_descriptions}
    files_to_process = [
        rel_path for rel_path in all_relative_paths
        if rel_path not in processed_rgb_paths
    ]
    logging.info(f"{len(files_to_process)} new images to process.")

    try:
        for batch in tqdm(batched(files_to_process, args.batch_size), 
                          total=-(len(files_to_process) // -args.batch_size),
                          desc="Generating Descriptions with LLaVA-Next"):
            
            image_batch, path_batch = [], []
            for relative_rgb_path in batch:
                full_rgb_path = os.path.join(args.dataset_dir, relative_rgb_path)
                if os.path.exists(full_rgb_path):
                    image_batch.append(Image.open(full_rgb_path).convert("RGB"))
                    path_batch.append(relative_rgb_path)
                else:
                    logging.warning(f"Skipping, image not found at: {full_rgb_path}")

            if not image_batch:
                continue

            try:
                prompt_batch = [args.prompt] * len(image_batch)
                inputs = processor(
                    text=prompt_batch, 
                    images=image_batch, 
                    return_tensors="pt", 
                    padding=True
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                
                decoded_texts = processor.batch_decode(outputs, skip_special_tokens=True)

                for i, full_text in enumerate(decoded_texts):
                    # The decoded text still contains the prompt, keep only the answer
                    description = full_text.split("ASSISTANT:")[1].strip()
                    all_descriptions.append({
                        "rgb_path": path_batch[i],
                        "description": description
                    })

            except Exception as e:
                logging.error(f"Error processing batch. First image in batch: {path_batch[0]}. Error: {e}")

    except KeyboardInterrupt:
        logging.warning("\nProcess interrupted by user. Saving progress...")

    finally:
        # Also save when interrupted, so the run can be resumed
        with open(output_file_path, 'w') as f:
            json.dump(all_descriptions, f, indent=4)
        logging.info(f"\nProcessing complete. {len(all_descriptions)} descriptions saved to {output_file_path}")

if __name__ == "__main__":
    main()

import torch
import torchvision.transforms as T
from PIL import Image
import os
import json
from tqdm import tqdm
import argparse
import logging
from itertools import islice
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Image preprocessing from the InternVL3 model card: each image is split into 448x448 tiles
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_tiles(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def batched(iterable, n):
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            return
        yield chunk

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate image captions using InternVL3.")
    
    parser.add_argument("--model_id", type=str, default="OpenGVLab/InternVL3-38B",
                        help="The Hugging Face model ID for InternVL3.")
    parser.add_argument("--dataset_dir", type=str, default="datasets/NYUDepthv2",
                        help="Path to the root of the dataset directory.")
    parser.add_argument("--input_files", nargs='+', default=['train.txt', 'test.txt'],
                    help="One or more input files. Defaults to ['train.txt', 'test.txt'].")
    parser.add_argument("--output_file", type=str, default="captions/NYUDepthv2/internvl3.json",
                        help="Path of the output JSON file to save descriptions.")
    parser.add_argument("--prompt", type=str, 
                        default="<image>\nWrite a strictly factual description of the image, focusing on object positions and spatial relations.",
                        help="The text prompt to guide the VLM. Must include '<image>'.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of images to process in a single batch.")
    parser.add_argument("--max_new_tokens", type=int, default=300,
                        help="Maximum number of new tokens to generate.")
    
    return parser.parse_args()

def load_model_and_tokenizer(model_id):
    """Load InternVL3 (bfloat16) and its tokenizer."""
    logging.info(f"Loading model: {model_id}")
    
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    
    logging.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

def main():
    args = parse_arguments()
    
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    
    output_file_path = args.output_file
    os.makedirs(os.path.dirname(output_file_path) or ".", exist_ok=True)

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
                          desc="Generating Descriptions with InternVL3"):
            
            path_batch = []
            pixel_values_batch = []
            num_patches_list = []

            for relative_rgb_path in batch:
                full_rgb_path = os.path.join(args.dataset_dir, relative_rgb_path)
                if os.path.exists(full_rgb_path):
                    pixel_values = load_image_tiles(full_rgb_path, max_num=12)
                    pixel_values_batch.append(pixel_values)
                    num_patches_list.append(pixel_values.size(0))
                    path_batch.append(relative_rgb_path)
                else:
                    logging.warning(f"Skipping, image not found at: {full_rgb_path}")

            if not path_batch:
                continue

            try:
                # The tiles of all images go into one tensor, num_patches_list tells batch_chat which belong together
                pixel_values = torch.cat(pixel_values_batch, dim=0).to(model.device, dtype=model.dtype)
                
                questions = [args.prompt] * len(path_batch)
                generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

                with torch.no_grad():
                    responses = model.batch_chat(
                        tokenizer, 
                        pixel_values,
                        num_patches_list=num_patches_list,
                        questions=questions,
                        generation_config=generation_config
                    )

                for i, description in enumerate(responses):
                    all_descriptions.append({
                        "rgb_path": path_batch[i],
                        "description": description.strip()
                    })

            except Exception as e:
                logging.error(f"Error processing batch. First image in batch: {path_batch[0]}. Error: {e}")

    except KeyboardInterrupt:
        logging.warning("\nProcess interrupted by user. Saving progress...")

    finally:
        with open(output_file_path, 'w') as f:
            json.dump(all_descriptions, f, indent=4)
        logging.info(f"\nProcessing complete. {len(all_descriptions)} descriptions saved to {output_file_path}")

if __name__ == "__main__":
    main()

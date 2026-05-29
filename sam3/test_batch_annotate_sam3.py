import os
import json
import torch
import typer
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from transformers import Sam3Processor, Sam3Model

# Load environment variables automatically
load_dotenv()

app = typer.Typer()

@app.command()
def process_dataset(
    input_dir: str = typer.Argument(..., help="Folder główny ze zdjęciami do przetworzenia"),
    query_dict_path: str = typer.Argument(..., help="Ścieżka do pliku queries.json z progami")
):
    """
    Advanced Batch SAM3 Processor.
    Uses a JSON file to run multiple prompts against images in a directory.
    Outputs RGB visual overlays, standard PNG masks, and raw .npy tensors for Grasp-Net.
    """
    
    # --- 1. Validation & Setup ---
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("Warning: HF_TOKEN environment variable not set in .env. Accessing model will fail.")
        raise typer.Exit(code=1)
    
    if not os.path.isdir(input_dir):
        print(f"Error: Directory not found: {input_dir}")
        raise typer.Exit(code=1)
        
    if not os.path.isfile(query_dict_path):
        print(f"Error: JSON file not found: {query_dict_path}")
        raise typer.Exit(code=1)

    with open(query_dict_path, 'r') as f:
        query_dict = json.load(f)

    for key, value in query_dict.items():
        if not isinstance(value, dict) or 'confidence' not in value or 'mask_confidence' not in value:
            print(f"Error: Invalid format in queries.json at key: {key}")
            raise typer.Exit(code=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware initialization... Using device: {device}")

    # --- 2. Load Model ---
    try:
        print("Loading SAM3 Model into VRAM...")
        model = Sam3Model.from_pretrained(
            "facebook/sam3", 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            token=HF_TOKEN
        ).to(device)
        processor = Sam3Processor.from_pretrained("facebook/sam3", token=HF_TOKEN)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise typer.Exit(code=1)

    # Output directory setup
    output_dir = os.path.join(input_dir, "sam3_annotated_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [p for p in Path(input_dir).iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    print(f"Found {len(image_paths)} valid images in {input_dir}.")

    # --- 3. Processing Loop ---
    query_colors = [
        (255, 0, 0),   # Red for query 1
        (0, 255, 0),   # Green for query 2
        (0, 0, 255),   # Blue for query 3
        (255, 255, 0), # Yellow for query 4 (if needed)
        (255, 0, 255)  # Magenta for query 5 (if needed)
    ]

    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {img_path.name}: {e}")
            continue
            
        combined_rgb = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        
        # Single loop over all queries
        for query_idx, (query, thresholds) in enumerate(query_dict.items()):
            confidence = thresholds.get("confidence", 0.5)
            mask_confidence = thresholds.get("mask_confidence", 0.5)

            inputs = processor(images=image, text=query, return_tensors="pt").to(device)
            for key in inputs:
                if inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(model.dtype)

            # --- INFERENCE (Only run once per query) ---
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=confidence,
                mask_threshold=mask_confidence,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            masks = results["masks"]
            
            # --- PROCESS RESULTS ---
            query_union = np.zeros((image.height, image.width), dtype=bool)
            combined_mask_grayscale = np.zeros((image.height, image.width), dtype=np.uint8)
            
            if len(masks) > 0:
                for idx, mask in enumerate(masks):
                    mask_np = mask.squeeze().cpu().numpy() > 0.5
                    # Add to boolean union for RGB visualization
                    query_union |= mask_np
                    # Add to grayscale map for individual object instances (1, 2, 3...)
                    combined_mask_grayscale[mask_np] = (idx % 255) + 1

            # 1. Update the RGB Visualization map
            color = query_colors[query_idx % len(query_colors)]
            combined_rgb[query_union] = color

            # 2. Save individual query mask as PNG (Grayscale)
            safe_query_name = query.replace(' ', '_')
            base_name = img_path.stem
            
            output_mask_png = os.path.join(output_dir, f"{base_name}_{safe_query_name}_mask.png")
            Image.fromarray(combined_mask_grayscale).save(output_mask_png)
            
            # 3. Save raw boolean tensor as .npy for Grasp-Net / ROS2 processing
            output_mask_npy = os.path.join(output_dir, f"{base_name}_{safe_query_name}_tensor.npy")
            np.save(output_mask_npy, query_union)

        # 4. Save the combined RGB visualization for the whole image
        output_rgb_path = os.path.join(output_dir, f"{base_name}_combined_queries_rgb.png")
        Image.fromarray(combined_rgb, mode="RGB").save(output_rgb_path)

if __name__ == "__main__":
    app()
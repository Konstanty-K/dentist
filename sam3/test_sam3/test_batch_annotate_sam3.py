import os
import json
import torch
import typer
import numpy as np
from PIL import Image, ImageDraw
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
    query_dict_path: str = typer.Argument(..., help="Ścieżka do pliku queries.json z progami"),
    show_labels: bool = typer.Option(False, "--show-labels", help="Rysuj ramki (bounding boxes), etykiety i % pewności na wizualizacji RGB")
):
    """
    Advanced Batch SAM3 Processor.
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

    output_dir = os.path.join(input_dir, "sam3_annotated_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [p for p in Path(input_dir).iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    print(f"Found {len(image_paths)} valid images in {input_dir}.")

    # --- 3. Processing Loop ---
    query_colors = [
        (255, 50, 50),   # Red
        (50, 255, 50),   # Green
        (50, 150, 255),  # Blue
        (255, 255, 50),  # Yellow
        (255, 50, 255)   # Magenta
    ]

    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {img_path.name}: {e}")
            continue
            
        # Create a transparent overlay for drawing masks and bounding boxes
        overlay_rgba = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_rgba)
        
        for query_idx, (query, thresholds) in enumerate(query_dict.items()):
            confidence = thresholds.get("confidence", 0.5)
            mask_confidence = thresholds.get("mask_confidence", 0.5)
            color_rgb = query_colors[query_idx % len(query_colors)]
            color_rgba = (*color_rgb, 120) # 120 is the alpha channel for semi-transparency

            inputs = processor(images=image, text=query, return_tensors="pt").to(device)
            for key in inputs:
                if inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(model.dtype)

            # --- INFERENCE ---
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=confidence,
                mask_threshold=mask_confidence,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            masks = results["masks"]
            scores = results.get("scores")
            
            query_union = np.zeros((image.height, image.width), dtype=bool)
            combined_mask_grayscale = np.zeros((image.height, image.width), dtype=np.uint8)
            
            if len(masks) > 0:
                for idx, mask in enumerate(masks):
                    mask_np = mask.squeeze().cpu().numpy() > 0.5
                    if not np.any(mask_np): continue # Skip empty masks
                    
                    query_union |= mask_np
                    combined_mask_grayscale[mask_np] = (idx % 255) + 1

                    # Draw Bounding Boxes and Labels if flag is enabled
                    if show_labels:
                        # Calculate bounding box from mask coordinates
                        y_coords, x_coords = np.where(mask_np)
                        x_min, x_max = int(x_coords.min()), int(x_coords.max())
                        y_min, y_max = int(y_coords.min()), int(y_coords.max())

                        # Draw Rectangle
                        draw.rectangle([x_min, y_min, x_max, y_max], outline=color_rgb, width=3)
                        
                        # Draw Label Background and Text
                        score_val = scores[idx].item() if scores is not None else 0.0
                        label_text = f"{query}: {score_val:.0%}"
                        
                        # Simple background for text visibility
                        text_bbox = draw.textbbox((x_min, y_min - 15), label_text)
                        draw.rectangle(text_bbox, fill=color_rgb)
                        draw.text((x_min, y_min - 15), label_text, fill=(255, 255, 255))

            # Draw the semi-transparent mask on the overlay
            if np.any(query_union):
                # Convert boolean numpy array to a PIL Image mask
                mask_image = Image.fromarray((query_union * 255).astype(np.uint8), mode='L')
                draw.bitmap((0, 0), mask_image, fill=color_rgba)

            # Save individual query mask (Grayscale)
            safe_query_name = query.replace(' ', '_')
            base_name = img_path.stem
            output_mask_png = os.path.join(output_dir, f"{base_name}_{safe_query_name}_mask.png")
            Image.fromarray(combined_mask_grayscale).save(output_mask_png)
            
            # Save raw boolean tensor
            output_mask_npy = os.path.join(output_dir, f"{base_name}_{safe_query_name}_tensor.npy")
            np.save(output_mask_npy, query_union)

        # Composite the original image with the overlay and save
        annotated_image = Image.alpha_composite(image.convert('RGBA'), overlay_rgba)
        output_rgb_path = os.path.join(output_dir, f"{base_name}_combined_queries_rgb.png")
        annotated_image.convert('RGB').save(output_rgb_path)

if __name__ == "__main__":
    app()
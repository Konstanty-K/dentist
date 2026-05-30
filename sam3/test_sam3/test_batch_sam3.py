import typer
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("Warning: HF_TOKEN environment variable is not set. Make sure your .env file is configured.")

app = typer.Typer()

def show_mask(mask, ax, random_color=False):
    """Applies a semi-transparent mask overlay on the given matplotlib axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

@app.command()
def batch_infer(
    input_dir: str = typer.Argument(..., help="Directory containing input images"),
    prompt: str = typer.Argument(..., help="Text prompt for SAM3 (e.g., 'surgical tool')"),
    output_dir: str = typer.Argument(..., help="Directory to save the output visualizations and raw masks"),
    threshold: float = typer.Option(0.5, help="Confidence threshold for object detection"),
    mask_threshold: float = typer.Option(0.5, help="Binarization threshold for the generated masks")
):
    """
    Batch processing script for SAM3. 
    Loads the model into VRAM once and processes a directory of images sequentially.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load model ONLY ONCE to prevent VRAM overflow and save time
    print("Loading SAM3 model into GPU memory...")
    try:
        model = Sam3Model.from_pretrained(
            "facebook/sam3", 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            token=HF_TOKEN
        ).to(device)
        processor = Sam3Processor.from_pretrained("facebook/sam3", token=HF_TOKEN)
    except Exception as e:
        print(f"Model loading error: {e}")
        raise typer.Exit(code=1)

    # 3. Retrieve list of valid images
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in valid_exts]
    
    print(f"Found {len(image_paths)} images to process.")

    # 4. Processing loop
    for img_path in image_paths:
        print(f"\n--- Processing: {img_path.name} ---")
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, text=prompt.strip(), return_tensors="pt").to(device)
            
            # Cast float32 inputs to model's dtype (float16 for GPU)
            for key in inputs:
                if inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(model.dtype)
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            results = processor.post_process_instance_segmentation(
                outputs, threshold=threshold, mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            
            masks = results['masks']
            num_objects = len(masks)
            
            # --- VISUALIZATION ---
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            axes[0].imshow(image); axes[0].set_title("Original"); axes[0].axis('off')
            axes[1].imshow(image); axes[1].set_title(f"Result for '{prompt}' ({num_objects} objects)"); axes[1].axis('off')
            
            masks_np = masks.cpu().numpy()
            for mask in masks_np:
                show_mask(mask, axes[1], random_color=True)
                
            plt.tight_layout()
            
            save_img_path = os.path.join(output_dir, f"sam3_{img_path.name}")
            plt.savefig(save_img_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig) # Closing the figure frees up RAM!
            
            # --- SAVE RAW MASKS (.npy) ---
            # Saving as NumPy arrays for downstream use (e.g., Grasp-Net calculations)
            raw_mask_path = os.path.join(output_dir, f"raw_{img_path.stem}.npy")
            np.save(raw_mask_path, masks_np)
            
            print(f"Saved visualization: {save_img_path}")
            print(f"Saved raw masks: {raw_mask_path}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            # Emergency VRAM cleanup on specific frame error to prevent cascade failures
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    app()
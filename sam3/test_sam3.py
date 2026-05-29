import typer
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import os

# Get the token from the environment variable (or set it manually here for testing)
HF_TOKEN = os.environ.get("HF_TOKEN")

# Check if the token was found (useful for error checking)
if not HF_TOKEN:
    print("Warning: HF_TOKEN environment variable not set. Accessing a gated model will fail.")

app = typer.Typer()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

@app.command()
def infer(
    image_path: str = typer.Argument(..., help="Path to the input image"),
    prompt: str = typer.Argument(..., help="Text prompt for segmentation"),
    output_path: str = typer.Option("output.png", help="Path to save the result"),
    threshold: float = typer.Option(0.5, help="Confidence threshold for predictions"),
    mask_threshold: float = typer.Option(0.5, help="Mask threshold")
):
    """
    Run SAM3 inference using Hugging Face Transformers with a text prompt.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and processor
    try:
        print("Loading model...")
        model = Sam3Model.from_pretrained(
            "facebook/sam3", 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            token=HF_TOKEN
        ).to(device)
        processor = Sam3Processor.from_pretrained("facebook/sam3", token=HF_TOKEN)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise typer.Exit(code=1)

    # Load Image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        raise typer.Exit(code=1)

    print(f"Processing image: {image_path} with prompt: '{prompt}'")

    try:
        # Prepare inputs
        inputs = processor(images=image, text=prompt.strip(), return_tensors="pt").to(device)
        
        # Cast float32 inputs to model dtype (fp16 if cuda)
        for key in inputs:
            if inputs[key].dtype == torch.float32:
                inputs[key] = inputs[key].to(model.dtype)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-processing
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        masks = results['masks']
        scores = results['scores']
        
        num_objects = len(masks)
        print(f"Found {num_objects} objects matching '{prompt}'")

        if num_objects == 0:
            print("No objects found. Try adjusting thresholds.")
            return

        # Visualization: Side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # 1. Original Image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # 2. Segmented Image
        axes[1].imshow(image)
        axes[1].set_title(f"Segmented: '{prompt}' ({num_objects} objects)")
        axes[1].axis('off')
        
        masks_np = masks.cpu().numpy()
        scores_np = scores.cpu().numpy()

        # Iterate over detected masks and overlay on the second subplot
        for i, mask in enumerate(masks_np):
            print(f"Mask {i+1}: Score {scores_np[i]:.4f}")
            show_mask(mask, axes[1], random_color=True)

        plt.tight_layout()
        
        # Save the side-by-side comparison
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Side-by-side result saved to {output_path}")

        # Try to display if environment supports it
        try:
            plt.show()
        except Exception:
            pass
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
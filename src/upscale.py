import os
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from super_image import PanModel, ImageLoader

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Batch Image Upscaler using Deep Learning (PanModel)."
    )
    
    # CLI Arguments
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the folder containing input images (.jpg, .png)."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./upscaled_output", 
        help="Path to the folder where upscaled images will be saved."
    )
    parser.add_argument(
        "--scale", 
        type=int, 
        default=4, 
        choices=[2, 3, 4], 
        help="Upscaling factor. Choose 2, 3, or 4 (Default: 4)."
    )
    parser.add_argument(
        "--force-cpu", 
        action="store_true", 
        help="Force the script to use CPU even if CUDA is available."
    )

    return parser.parse_args()

def load_model(scale, force_cpu=False):
    """
    Loads the PAN (Pixel Attention Network) model.
    """
    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    print(f"Initializing model on device: {device.upper()}")
    
    try:
        # Using the pre-trained BAM (Balanced Attention Mechanism) model
        model = PanModel.from_pretrained('eugenesiow/pan-bam', scale=scale)
        model = model.to(device)
        model.eval() # Set to inference mode
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def process_images(input_folder, output_folder, model, device):
    """
    Iterates through the input folder and upscales images.
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Filter valid image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"No valid images found in {input_folder}")
        return

    print(f"Found {len(image_files)} images. Starting upscale process...")

    # Progress bar loop
    for filename in tqdm(image_files, desc="Upscaling"):
        input_path = os.path.join(input_folder, filename)
        output_filename = f"upscaled_x{model.config.scale}_{filename}"
        output_path = os.path.join(output_folder, output_filename)

        try:
            # Load Image
            image = Image.open(input_path).convert('RGB')
            
            # Prepare Input
            inputs = ImageLoader.load_image(image).to(device)
            
            # Inference (No Gradients needed)
            with torch.no_grad():
                preds = model(inputs)
            
            # Save Image
            ImageLoader.save_image(preds, output_path)
            
        except Exception as e:
            print(f"\n[Error] Failed to process {filename}: {e}")

    print(f"\nProcessing complete. Images saved to: {output_folder}")

def main():
    args = parse_arguments()

    # check input existence
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist.")
        return

    # Load resources
    model, device = load_model(args.scale, args.force_cpu)

    # Run pipeline
    process_images(args.input, args.output, model, device)

if __name__ == "__main__":
    main()
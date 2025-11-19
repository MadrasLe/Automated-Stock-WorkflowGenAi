import os
import torch
import gc
import re
import argparse
import time
from unsloth import FastLanguageModel
from diffusers import DiffusionPipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM + SDXL Image Generation Pipeline")
    
    # Input arguments
    parser.add_argument("--topic", type=str, required=True, help="Description of the topic/theme for the image prompts (e.g., 'Cyberpunk City', 'Nature Photography').")
    parser.add_argument("--count", type=int, default=5, help="Number of prompts/images to generate (default: 5).")
    parser.add_argument("--steps", type=int, default=40, help="Number of inference steps for SDXL (default: 40).")
    parser.add_argument("--output", type=str, default="./output_images", help="Directory to save generated images.")
    
    return parser.parse_args()

def generate_prompts_with_llama(topic, count):
    print(f"\n[Phase 1] Initializing Llama 3 to generate {count} prompts about '{topic}'...")
    
    max_seq_length = 4096
    dtype = None
    load_in_4bit = True

    # Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/llama-3-8b-Instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    # Construct Prompt
    messages = [
        {
            "role": "system",
            "content": "You are an expert prompt engineer for image AIs like SDXL. Your mission is to create detailed, vivid, and photorealistic prompts perfect for stock photos. Output ONLY the numbered list of prompts, without any introduction or conclusion. Your prompts should be optimized for SDXL's understanding of natural language."
        },
        {
            "role": "user",
            "content": f"Create a list of {count} highly detailed prompts for stock photos. Focus on: {topic}. Include details on lighting, camera settings (like aperture, lens), and mood. Format it as a numbered list."
        },
    ]

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs_tensor = model.generate(**inputs, max_new_tokens=2048, use_cache=True, eos_token_id=tokenizer.eos_token_id)
        text_output = tokenizer.decode(outputs_tensor[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    # Parse output
    prompts_list = []
    for line in text_output.strip().split('\n'):
        cleaned_line = re.sub(r'^\d+\.\s*', '', line).strip()
        if cleaned_line:
            prompts_list.append(cleaned_line)

    print("Prompts generated successfully.")
    
    # Clean up Memory
    del model, tokenizer, inputs, outputs_tensor
    gc.collect()
    torch.cuda.empty_cache()
    print("[Phase 1 Complete] Llama 3 unloaded. GPU memory released.\n")
    
    return prompts_list

def generate_images_with_sdxl(prompts, output_dir, steps):
    print("[Phase 2] Loading SDXL Base and Refiner models...")
    
    try:
        # Load Base
        pipe_base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")

        # Load Refiner (Reusing components to save VRAM)
        pipe_refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipe_base.text_encoder_2,
            vae=pipe_base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")
        
        print("SDXL Models loaded to GPU.")
        
        os.makedirs(output_dir, exist_ok=True)
        high_noise_frac = 0.8
        
        for i, prompt in enumerate(prompts):
            sanitized_prompt = (prompt[:75] + '..') if len(prompt) > 75 else prompt
            print(f"Generating Image {i+1}/{len(prompts)}: {sanitized_prompt}")
            
            # Base generation
            image = pipe_base(
                prompt=prompt,
                num_inference_steps=steps,
                denoising_end=high_noise_frac,
                output_type="latent",
            ).images
            
            # Refiner generation
            image = pipe_refiner(
                prompt=prompt,
                num_inference_steps=steps,
                denoising_start=high_noise_frac,
                image=image,
            ).images[0]
            
            timestamp = int(time.time())
            filename = f"sdxl_{timestamp}_{i+1}.png"
            save_path = os.path.join(output_dir, filename)
            image.save(save_path)
            print(f"  Saved: {save_path}")

        print("\n[Success] All images generated successfully.")

    except Exception as e:
        print(f"\n[Error] An error occurred during image generation: {e}")
        print("Ensure you have sufficient VRAM (approx 16GB recommended for this pipeline without CPU offload).")

def main():
    args = parse_arguments()
    
    # Check for CUDA
    if not torch.cuda.is_available():
        print("Error: CUDA GPU is required for this script.")
        return

    # Phase 1: Generate Prompts
    prompts = generate_prompts_with_llama(args.topic, args.count)
    
    if not prompts:
        print("Error: No prompts were generated.")
        return

    print("--- Selected Prompts ---")
    for idx, p in enumerate(prompts):
        print(f"{idx+1}. {p}")
    print("------------------------\n")

    # Phase 2: Generate Images
    generate_images_with_sdxl(prompts, args.output, args.steps)

if __name__ == "__main__":
    main()
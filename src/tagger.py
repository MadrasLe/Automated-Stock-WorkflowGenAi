import os
import csv
import argparse
import spacy
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Ensure spaCy model is available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[System] Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Automated Image Tagger using Florence-2 (Exports to CSV & TXT)."
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the folder containing images to analyze."
    )
    parser.add_argument(
        "--categories", 
        nargs='*', 
        default=[], 
        help="List of up to 3 categories (e.g., --categories Technology Business Abstract)."
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        default=4, 
        help="Number of concurrent threads for processing (Default: 4)."
    )
    parser.add_argument(
        "--output-name", 
        type=str, 
        default="dreamstime_metadata", 
        help="Base name for the output files (without extension)."
    )
    parser.add_argument(
        "--force-cpu", 
        action="store_true", 
        help="Force usage of CPU instead of GPU."
    )

    return parser.parse_args()

class ImageAnalyzer:
    def __init__(self, image_folder, categories, num_threads, force_cpu=False):
        self.image_folder = image_folder
        # Ensure we have exactly 3 category slots, filling empty ones with empty strings
        self.categories = (categories + ['', '', ''])[:3] 
        self.num_threads = num_threads
        
        self.device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        print(f"[Init] Running on device: {self.device.upper()}")
        
        self.processor, self.model = self._load_models()

    def _load_models(self):
        print("[Init] Loading Florence-2-base model...")
        model_id = 'microsoft/Florence-2-base'
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()
        model.to(self.device)
        
        # Load processor
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        return processor, model

    def _generate_keywords(self, description, max_keywords=50, min_length=3):
        """Extracts keywords from the description using NLP."""
        doc = nlp(description.lower())
        keywords = set()
        
        # Extract individual tokens (Nouns, Adjectives, Proper Nouns)
        for token in doc:
            if token.pos_ in ["NOUN", "ADJ", "PROPN"] and not token.is_stop and len(token.text) >= min_length:
                keywords.add(token.text)
        
        # Extract noun chunks
        for chunk in doc.noun_chunks:
            clean_chunk = ' '.join(token.text for token in chunk if not token.is_stop and token.pos_ in ["NOUN", "ADJ", "PROPN"])
            if len(clean_chunk) > min_length:
                keywords.add(clean_chunk)
                
        return ', '.join(list(keywords)[:max_keywords])

    def analyze_single_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            task_prompt = "<MORE_DETAILED_CAPTION>"
            
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
            
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=image.size
            )
            
            description = parsed_answer.get(task_prompt, "Description generation failed.")
            
            # Create a title from the first few words
            title = ' '.join(description.split()[:10]).strip()
            keywords = self._generate_keywords(description)
            
            return {
                'description': description.capitalize(),
                'title': title.capitalize(),
                'keywords': keywords
            }
            
        except Exception as e:
            print(f"\n[Error] Failed to analyze {os.path.basename(image_path)}: {e}")
            return None

    def process_batch(self, filename):
        full_path = os.path.join(self.image_folder, filename)
        analysis = self.analyze_single_image(full_path)
        return filename, analysis

    def run(self, output_base_name):
        valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
        images = [f for f in os.listdir(self.image_folder) if f.lower().endswith(valid_exts)]
        
        if not images:
            print(f"[Error] No images found in '{self.image_folder}'")
            return

        print(f"[Process] Found {len(images)} images. Starting analysis...")
        
        data_for_csv = []
        txt_filename = f"{output_base_name}.txt"
        csv_filename = f"{output_base_name}.csv"

        with open(txt_filename, "w", encoding="utf-8") as txt_file:
            # Use ThreadPool for parallel processing (loading images/pre-processing)
            # Note: Actual inference is sequential on GPU usually, but threading helps IO
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {executor.submit(self.process_batch, img): img for img in images}
                
                for future in tqdm(futures, total=len(images), desc="Analyzing"):
                    filename, analysis = future.result()
                    
                    if analysis is None:
                        continue
                    
                    # Prepare Metadata Record
                    record = {
                        'Filename': filename,
                        'Image Name': f"{analysis['title']}",
                        'Descriptions': analysis['description'],
                        'Keywords': analysis['keywords'],
                        'Category 1': self.categories[0],
                        'Category 2': self.categories[1],
                        'Category 3': self.categories[2],
                        'Free': 0, 
                        'W-EL': 1, 
                        'P-EL': 1, 
                        'SR-EL': 1, 
                        'SR-Price': 100
                    }
                    
                    # Write to TXT immediately
                    for key, value in record.items():
                        txt_file.write(f"{key}: {value}\n")
                    txt_file.write("\n")
                    
                    # Collect for CSV
                    data_for_csv.append(record)

        # Write CSV
        if data_for_csv:
            print(f"[Export] Writing CSV file: {csv_filename}")
            with open(csv_filename, "w", encoding="utf-8", newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=data_for_csv[0].keys())
                writer.writeheader()
                writer.writerows(data_for_csv)

        print(f"\n[Success] Processing complete.")
        print(f" -> TXT: {os.path.abspath(txt_filename)}")
        print(f" -> CSV: {os.path.abspath(csv_filename)}")

def main():
    args = parse_arguments()
    
    if not os.path.isdir(args.input):
        print(f"Error: The folder '{args.input}' does not exist.")
        return

    analyzer = ImageAnalyzer(
        image_folder=args.input,
        categories=args.categories,
        num_threads=args.threads,
        force_cpu=args.force_cpu
    )
    
    analyzer.run(args.output_name)

if __name__ == "__main__":
    main()
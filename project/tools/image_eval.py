# Acknowledgement, Hugging Face heavily influenced the creation of this code
# Model Link: https://huggingface.co/docs/transformers/main/en/model_doc/llava_onevision

# general dependencies
import os
import sys
import time
import re
import torch
import pandas as pd
from PIL import Image

# transformer dependencies
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from prompts import Prompts
from parse_response import ResponseParser


class ImageEval:
    def __init__(self, cuda_number=0, prompt_idx=None):
        print("\nInitializing ImageEval!")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.normpath(
            os.path.join(script_dir, "..", "results", "image")
        )

        self.set_prompt_index(prompt_idx)

        self.device = f"cuda:{cuda_number}"
        self.start_time = time.time()
        self.model, self.processor = self.GetModelAndProcessor()

    def GetModelAndProcessor(self):
        model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        processor = AutoProcessor.from_pretrained(model_id)

        return model, processor

    def set_prompt_index(self, prompt_idx):
        self.prompt_idx = prompt_idx
        if prompt_idx is None:
            self.text_prompts = None
            self.prompt_options = None
        else:
            self.text_prompts = Prompts.GetPromptList(prompt_idx)
            self.prompt_options = Prompts.GetPromptOptions(prompt_idx)

    def find_images_in_folder(self, directory):
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
        image_paths = []

        # Walk through the directory and its subdirectories
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Check if the file is an image
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))

        return sorted(image_paths)

    def extract_number_from_string(self, input_string):
        match = re.search(r"(\d+)", input_string)
        if match:
            return match.group(0)
        return None

    def extract_prefix_from_string(self, input_string):
        match = re.search(r"video\\([^\\]+)", input_string)
        if match:
            return match.group(1)
        return None

    def GetSampleData(self, dir, path):
        img_path = path.replace(dir, "")
        vid_path = img_path.replace("images", "video")
        number = self.extract_number_from_string(img_path)
        action = self.extract_prefix_from_string(img_path)
        return action, number, img_path, vid_path

    def inference(self, image):
        self.conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": Prompts.GetPrompt(self.text_prompts)},
                ],
            },
        ]
        chat = self.processor.apply_chat_template(
            self.conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[chat], images=[image], padding=True, return_tensors="pt"
        ).to(self.device)

        # generate outputs
        output_ids = self.model.generate(
            **inputs, pad_token_id=151645, max_new_tokens=256
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        return output_text.replace("\n", " ").replace(",", ";")

    def eval(self, image_directory, output_directory="default"):
        output_directory = os.path.join(self.results_dir, output_directory)
        os.makedirs(output_directory, exist_ok=True)
        output_filename = os.path.join(output_directory, "image_results.csv")

        parser = ResponseParser(None, self.prompt_idx)
        paths = self.find_images_in_folder(image_directory)
        for path in paths:
            action, number, img_path, vid_path = self.GetSampleData(
                image_directory, path
            )
            image = Image.open(path)
            response = "ERROR"

            try:
                response = self.inference(image)
                answers = parser.extract_answers(response)
                data = {
                    "action": action,
                    "sample": number,
                    "imgPath": img_path,
                    "vidPath": vid_path,
                    "dataset": "default",
                    "response": response,
                    "one_person": answers[0],
                    "face_visible": answers[1],
                    "gender": answers[2],
                    "age": answers[3],
                    "race": answers[4],
                }

            except Exception as e:
                print(f"An error occurred: {e}")
                data = {
                    "action": action,
                    "sample": number,
                    "imgPath": img_path,
                    "vidPath": vid_path,
                    "dataset": "default",
                    "response": response,
                    "one_person": "ERROR",
                    "face_visible": "ERROR",
                    "gender": "ERROR",
                    "age": "ERROR",
                    "race": "ERROR",
                }

            # Convert single data dictionary to a DataFrame
            df = pd.DataFrame([data])

            # Append to CSV, adding header only if file doesn't exist
            df.to_csv(
                output_filename,
                mode="a",
                header=not pd.io.common.file_exists(output_filename),
                index=False,
            )

        print("ImageEval Finished!")

        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"Elapsed time: {elapsed_time} seconds\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        torch.cuda.empty_cache()


def main():
    # read the parameters from the command line
    prompt_idx = int(sys.argv[1])
    images_dir = sys.argv[2]

    output_dir = "default"

    if len(sys.argv) > 3:
        output_dir = sys.argv[3]

    image_eval = ImageEval()
    image_eval.set_prompt_index(prompt_idx)
    image_eval.eval(image_directory=images_dir, output_directory=output_dir)


if __name__ == "__main__":
    main()

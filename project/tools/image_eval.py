# Acknowledgement, Hugging Face heavily influenced the creation of this code
# Model Link: https://huggingface.co/docs/transformers/main/en/model_doc/llava_onevision

# general dependencies
import os
import sys
import time
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from PIL import Image

# transformer dependencies
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from prompts import Prompts
from gen_plots import PlotGeneration

class ImageEval:
    def __init__(self, cuda_number=0):
        print("\nInitializing ImageEval!")

        # print(f"Numpy version: {np.__version__}") # 2.2.4

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.normpath(os.path.join(script_dir, "..", "results", "image"))

        self.device = f"cuda:{cuda_number}"
        self.text_prompts = {}
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
    
    def find_images_in_folder(self, directory):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        image_paths = []

        # Walk through the directory and its subdirectories
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Check if the file is an image
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))

        return sorted(image_paths)
    
    def extract_number_from_string(self, input_string):
        match = re.search(r'(\d+)', input_string)
        if match:
            return match.group(0)
        return None

    def extract_prefix_from_string(self, input_string):
        match = re.search(r'video\\([^\\]+)', input_string)
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
                    {"type": "text", "text": Prompts().GetPrompt(self.text_prompts)},
                    ],
            },
        ]
        chat = self.processor.apply_chat_template(self.conversation, add_generation_prompt=True)

        inputs = self.processor(
            text=[chat],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # generate outputs
        output_ids = self.model.generate(**inputs, pad_token_id=151645, max_new_tokens=256)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        # self.conversation = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "path": image},
        #             {"type": "text", "text": Prompts().GetPrompt(self.text_prompts)},
        #             ],
        #     },
        # ]
        # try:
        #     inputs = self.processor.apply_chat_template(
        #         self.conversation,
        #         add_generation_prompt=True,
        #         padding=True,
        #         tokenize=True,
        #         return_dict=True,
        #         return_tensors="pt"
        #     )
        #     for key, value in inputs.items():
        #         print(key, value.shape)
        #     inputs = inputs.to(self.device, torch.float16)
        # except Exception as e:
        #     print(f"processor: {e}")
        # output_ids = self.model.generate(**inputs, pad_token_id=151645)
        # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        # output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        return output_text.replace('\n', ' ').replace(',', ';')
    
    def extract_answers(self, model_output):
        allowed_options = {
            'one_person': self.prompt_options[1],
            'face_visible': self.prompt_options[2],
            'gender': self.prompt_options[3],
            'age': self.prompt_options[4],
            'race': self.prompt_options[5],
        }

        keys_order = ['one_person', 'face_visible', 'gender', 'age', 'race']
        segments = re.findall(r"(\d+\.\s*.*?)(?=\d+\.\s*|$)", model_output, flags=re.DOTALL)

        if len(segments) == 0:
            segments = re.findall(r'\S+', model_output)
        if len(segments) < len(self.prompt_options) - 1:
            segments += [""] * (5 - len(segments))
        
        answers = []
        has_error = False

        for idx, key in enumerate(keys_order):
            seg = segments[idx].strip().lower()

            # Remove the questions from seg
            for phrase in self.text_prompts:
                if phrase is not None:
                    phrase = phrase.lower()
                    if phrase in seg:
                        seg = re.sub(re.escape(phrase), "", seg, flags=re.IGNORECASE).strip()

            # Remove any parenthetical content which may contain options
            seg = re.sub(r"\(option[^)]*\)", "", seg).strip()

            # If a colon (:) exists, assume the answer is after it
            if ":" in seg:
                candidate = seg.split(":")[-1].strip()
            else:
                candidate = seg.strip()

            # print(f"A: {candidate}")

            # Get allowed options for the key
            options = allowed_options.get(key, [])

            # First, check for an exact match
            exact_matches = [option.lower() for option in options if candidate == option.lower()]
            if len(exact_matches) == 1:
                found_option = exact_matches[0]
                # print(f"B: {found_option}")
            else:
                # Otherwise, perform a word-boundary search with the allowed options.
                # Sort options in descending order by length to ensure longer phrases (like "not visible")
                # are considered before shorter ones.
                options_sorted = sorted(options, key=lambda x: len(x), reverse=True)
                matches = []
                for option in options_sorted:
                    option = option.lower()
                    # print(f"C: {option}")
                    pattern = rf'(?:(?<=^)|(?<=[\s\.,;:!?\-\(\)])){re.escape(option)}(?=$|[\s\.,;:!?\-\(\)])'
                    if re.search(pattern, candidate):
                        accounted_for = False
                        for m in matches:
                            if " " + option + " " in " " + m + " ":
                                accounted_for = True
                        if not accounted_for:
                            matches.append(option)
                            # print(f"D: {option}")
                if len(matches) == 1:
                    found_option = matches[0]
                    # print(f"E: {found_option}")
                else:
                    found_option = None

            # Sort options in descending order of length to check longer phrases first
            # options_sorted = sorted(options, key=lambda x: len(x), reverse=True)

            # # Find all options matching as separate words (use boundaries)
            # matches = []
            # for option in options_sorted:
            #     option = option.lower()
                
            #     # Pattern that only matches when the option appears as a distinct word.
            #     pattern = rf'(?:(?<=^)|(?<=[\s\.,;:!?\-])){re.escape(option)}(?=$|[\s\.,;:!?\-])'
            #     if re.search(pattern, candidate):
            #         matches.append(option)

            # # If candidate contains more than one valid option or none, record an error
            # if len(matches) != 1 or candidate == "":
            #     answers.append("ERROR")
            #     has_error = True
            # else:
            #     answers.append(matches[0])

            # # Compare the candidate against the allowed options for this key
            # found_option = None
            # for option in allowed_options.get(key, []):
            #     option = option.lower()
            #     if re.search(rf'(?:(?<=^)|(?<=[\s\.,;:!\?\-])){re.escape(option)}(?=$|[\s\.,;:!\?\-])', candidate.lower()):
            #         found_option = option
            #         break
            
            if found_option is None or candidate == "":
                answers.append("ERROR")
                has_error = True
            else:
                answers.append(found_option)

        if has_error:
            print(f"Model output has an ERROR: {model_output}")
        
        return answers

    def process_csv(self, file):
        df = pd.read_csv(file)
        
        # Define the list of columns to update in the desired order.
        columns_to_update = ['one_person', 'face_visible', 'gender', 'age', 'race']

        def update_row(row):
            # Call the extract_answers method on the 'response' column.
            answers = self.extract_answers(row['response'])
            
            # Update each corresponding column with the respective answer.
            for col, answer in zip(columns_to_update, answers):
                row[col] = answer
            return row

        # Apply the update_row function to each row in the DataFrame.
        df = df.apply(update_row, axis=1)
        
        # Write the updated DataFrame back to the CSV file.
        df.to_csv(file, index=False)

    def eval(self, image_directory, text_prompts, prompt_options, output_directory="default"):
        output_directory = os.path.join(self.results_dir, output_directory)
        os.makedirs(output_directory, exist_ok=True)
        output_filename = os.path.join(output_directory, "image_results.csv")

        self.text_prompts = text_prompts
        self.prompt_options = prompt_options

        # print(self.extract_answers("1. Number of People: yes / no (Answer: yes) 2. Face Visibility: visible / not visible (Answer: visible) 3. Gender: female / other (Answer: female) 4. Age Group: adult / middle-aged (Answer: adult) 5. Race/Ethnicity: Indian / Southeast Asian (Answer: Indian)"))
        self.process_csv(output_filename)
        self.show_results(output_filename, output_directory)

        return
    
        i = 0
        paths = self.find_images_in_folder(image_directory)
        for path in paths:
            action, number, img_path, vid_path = self.GetSampleData(image_directory, path)
            image = Image.open(path)
            response="ERROR"

            try:
                response = self.inference(image)
                answers = self.extract_answers(response)
                data = {
                    'action': action,
                    'sample': number,
                    'imgPath': img_path,
                    'vidPath': vid_path,
                    'dataset': "default",
                    'response': response,
                    'one_person': answers[0],
                    'face_visible': answers[1],
                    'gender': answers[2],
                    'age': answers[3],
                    'race': answers[4],
                }
                i = i  +  1

            except Exception as e:
                print(f"An error occurred: {e}")
                data = {
                    'action': action,
                    'sample': number,
                    'imgPath': img_path,
                    'vidPath': vid_path,
                    'dataset': "default",
                    'response': response,
                    'one_person': "ERROR",
                    'face_visible': "ERROR",
                    'gender': "ERROR",
                    'age': "ERROR",
                    'race': "ERROR",
                }

            # Convert single data dictionary to a DataFrame
            df = pd.DataFrame([data])

            # Append to CSV, adding header only if file doesn't exist
            df.to_csv(output_filename, mode='a', header=not pd.io.common.file_exists(output_filename), index=False)
        
        print("ImageEval Finished!")

        self.show_results(output_filename, output_directory)

        # print time
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

    prompts = Prompts().GetPromptList(prompt_idx)
    prompt_options = Prompts().GetPromptOptions(prompt_idx)
    
    image_eval = ImageEval()
    image_eval.eval(
        image_directory=images_dir,
        text_prompts=prompts,
        prompt_options=prompt_options,
        output_directory=output_dir
    )

if __name__ == "__main__":
    main()
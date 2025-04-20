# Acknowledgement, Hugging Face heavily influenced the creation of this code
# Model Link: https://huggingface.co/docs/transformers/main/model_doc/qwen2_vl

# general dependencies
import os
import sys
import time
import torch
import pandas as pd

# transformer dependencies
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# Import Prompts
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)
from tools.prompts import Prompts


class Qwen2_5VL:
    def __init__(self, cuda_number=0):
        print("\nInitializing Qwen2_5VL!")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.normpath(
            os.path.join(script_dir, "..", "results", "video")
        )

        self.device = f"cuda:{cuda_number}"
        self.start_time = time.time()
        self.model, self.processor = self.GetModelAndProcessor()

    def GetModelAndProcessor(self):
        model = 0
        processor = 0

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        processor = AutoProcessor.from_pretrained(model_id)

        return model, processor

    def eval(
        self,
        start_idx,
        end_idx,
        bias_data_path,
        dataset_folder,
        text_prompt,
        output_directory="default",
        dataset_tag="default",
    ):
        bias_data = pd.read_csv(bias_data_path)

        output_directory = os.path.join(self.results_dir, output_directory)
        os.makedirs(output_directory, exist_ok=True)
        output_filename = os.path.join(output_directory, "qwen2_5_vl_output.csv")

        try:
            for idx in range(start_idx, end_idx + 1):
                sample = bias_data.iloc[idx]

                if sample["dataset"] != dataset_tag:
                    continue

                vid_path = dataset_folder + sample["vidPath"]

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": vid_path},
                            {"type": "text", "text": text_prompt},
                        ],
                    },
                ]

                inputs = self.processor.apply_chat_template(
                    conversation,
                    num_frames=4,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.device, torch.float16)

                out = self.model.generate(**inputs, pad_token_id=151645)
                output = self.processor.batch_decode(
                    out, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

                data = {
                    "action": sample["action"],
                    "sample": sample["sample"],
                    "imgPath": sample["imgPath"],
                    "vidPath": sample["vidPath"],
                    "dataset": sample["dataset"],
                    "response": output,
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

        except Exception as e:
            print(f"An error occurred: {e}")

        print("Qwen2_5VL Finished!")

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
    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    bias_data_path = sys.argv[3]
    dataset_folder = sys.argv[4]

    prompt_idx = 0
    output_directory = "default"
    dataset_tag = "default"
    cuda_number = 0

    if len(sys.argv) > 5:
        prompt_idx = int(sys.argv[5])

    if len(sys.argv) > 6:
        output_directory = sys.argv[6]

    if len(sys.argv) > 7:
        dataset_tag = sys.argv[7]

    if len(sys.argv) > 8:
        cuda_number = sys.argv[8]

    prompt = Prompts.GetPrompt(prompt_idx)

    qwen_2_5_vl = Qwen2_5VL(cuda_number=cuda_number)
    qwen_2_5_vl.eval(
        start_idx=start_idx,
        end_idx=end_idx,
        bias_data_path=bias_data_path,
        dataset_folder=dataset_folder,
        text_prompt=prompt,
        output_directory=output_directory,
        dataset_tag=dataset_tag,
    )


if __name__ == "__main__":
    main()

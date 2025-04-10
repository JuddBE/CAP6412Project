# Acknowledgement, Hugging Face heavily influenced the creation of this code
# Model Link: https://huggingface.co/docs/transformers/main/en/model_doc/video_llava

# general dependencies
import os
import av
import sys
import time
import torch
import shutil
import pandas as pd
import numpy as np

# transformer dependencies
import transformers
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor, BitsAndBytesConfig

# Import Prompts
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)
from tools.prompts import Prompts

class VideoLlava:
    def __init__(self, cuda_number=0):
        print("\nInitializing VideoLlava!")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.normpath(os.path.join(script_dir, "..", "results", "video"))

        self.device = f"cuda:{cuda_number}"
        self.start_time = time.time()
        self.model, self.processor = self.GetModelAndProcessor()

    def read_video_pyav(self, container, indices):
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def GetModelAndProcessor(self):        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
            
        model_id = "LanguageBind/Video-LLaVA-7B-hf"
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            model_id, quantization_config=quantization_config, torch_dtype=torch.float16
        ).to(self.device)
        processor = VideoLlavaProcessor.from_pretrained(model_id)

        return model, processor

    def eval(self, start_idx, end_idx, bias_data_path, dataset_folder, text_prompt, output_directory="default", dataset_tag="default"):
        bias_data = pd.read_csv(bias_data_path)

        output_directory = os.path.join(self.results_dir, output_directory)
        os.makedirs(output_directory, exist_ok=True)
        output_filename = os.path.join(output_directory, "video_llava_output.csv")
        
        try:
            for idx in range(start_idx, end_idx + 1):
                sample = bias_data.iloc[idx]

                if sample["dataset"] != dataset_tag:
                    continue

                vid_path = dataset_folder + sample['vidPath']

                container = av.open(vid_path)
                total_frames = container.streams.video[0].frames
                indices = np.arange(0, total_frames, total_frames / 8).astype(int)
                video = self.read_video_pyav(container, indices)
                
                prompt = "USER: <video>\n" + text_prompt + " ASSISTANT:"
                inputs = self.processor(
                    text=prompt, videos=video, return_tensors="pt"
                ).to(self.device)
                
                out = self.model.generate(**inputs, pad_token_id=151645)
                output = self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                data = {
                    'action': sample['action'],
                    'sample': sample['sample'],
                    'imgPath': sample['imgPath'],
                    'vidPath': sample['vidPath'],
                    'dataset': sample['dataset'],
                    'response': output
                }

                # Convert single data dictionary to a DataFrame
                df = pd.DataFrame([data])
            
                # Append to CSV, adding header only if file doesn't exist
                df.to_csv(output_filename, mode='a', header=not pd.io.common.file_exists(output_filename), index=False)
                
        except Exception as e:
            print(f"An error occurred: {e}")
        
        print("VideoLlava Finished!")
        
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

    prompts = Prompts()
    prompt = prompts.GetPrompt(prompt_idx)
    
    video_llava = VideoLlava(cuda_number=cuda_number)
    video_llava.eval(
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
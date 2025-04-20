import os
import sys

# Import Model Files
from one_vision_eval import OneVision
from llava_next_video_eval import LlavaNextVision
from video_llava_eval import VideoLlava
from qwen2_vl_eval import Qwen2VL
from qwen2_5_vl_eval import Qwen2_5VL

# Import Prompts
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)
from tools.prompts import Prompts


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

    model_classes = [
        lambda: OneVision(cuda_number=cuda_number),
        lambda: LlavaNextVision(cuda_number=cuda_number),
        lambda: VideoLlava(cuda_number=cuda_number),
        lambda: Qwen2VL(cuda_number=cuda_number),
        lambda: Qwen2_5VL(cuda_number=cuda_number),
    ]

    prompt = Prompts.GetPrompt(prompt_idx)

    for model in model_classes:
        with model() as evaluator:
            evaluator.eval(
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

# HAA500-B
## Project Overview
This is a project taking the existing HAA500 (v1.1) dataset and augmenting it for exploring bias. We introduce annotation data for the actions using GPT-4 and annotation data for videos using LLaVA-OneVision.
Special thanks to the creators of the original HAA500 dataset.

HAA500 Dataset Link: https://www.cse.ust.hk/haa/

HAA500 Paper Link: https://arxiv.org/abs/2009.05224 
 
## Contents Overview
**conda-env:** Contains conda environment requirements

**eval:** Evaluation scripts for the benchmark

**generation:** Scripts for generating clips from new data and combining files

**notebooks:** Jupyter notebooks for evaluating results on the benchmark and generating action bias data

**results:** All of the results generated for the project

**tools:** Helpful tools used by various parts of the project

**annotation-app:** Manual annotation app used for generating annotation data via human experts

**action_bias_data.csv:** The bias data generated for the actions in HAA500

**added_data.csv:** The new samples of data added for this project

**combined.csv:** The combined annotations used as ground truth for evaluation

## How To Use
In the *project/eval* directory, there are Python scripts for evaluating each model. These scripts can be ran like so:

*`python <script name>.py <start index> <end index> <bias data path> <dataset folder> [<prompt index>=0] [<output directory>=”default”] [<dataset tag>=”default”] [<cuda number>=0]`*

`All items between < and > are placeholders for the corresponding value. Items between [ and ] are optional with their default values coming after the ‘=’. A brief description of each item can be found below.`

* **`<script name>:`** The name of the script. For example, *one_vision_eval*

* **`<start index>:`** The start index for the data to be evaluated. For example, *0*

* **`<end index>:`** The end index for the data to be evaluated. For example, *10000*

* **`<bias data path>:`** The path to the CSV file which contains the paths of the videos to be evaluated. For example, *combined.csv*

* **`<dataset folder>:`** The path to the folder containing the videos in dataset. For example, *haa500_v1_1*

* **`<prompt index>:`** The index of the prompt to be supplied to the model. These prompts are found in *project/tools/prompts.py*. For example, *0*

* **`<output directory>:`** The name of the directory to store the output CSV file. This directory is relative to *project/results/video*. For example, a value of *default* would use the directory *project/results/video/default*

* **`<dataset tag>:`** The dataset tag of the data to be used for evaluation. This assumes there is a *dataset* column in the bias data CSV file. For example, a value of *default* will only evaluate data from the bias data CSV file with a dataset value of *default*

* **`<cuda number>:`** The GPU number to be used for loading the models and performing inference. For example, *0* (this will typically be the case with single GPU systems)

To streamline evaluation, there also exists the *main_eval.py* script, which accepts the same arguments as the other evaluation scripts, and will run all of the other evaluation scripts.

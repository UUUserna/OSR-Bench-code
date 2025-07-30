# OSR-Bench

OSR-Bench code

## Project Structure

```
OSR-Bench
├─data/                      # Dataset directory
│  ├─DeepPanoContext/       # DeepPanoContext dataset samples
│  └─ReplicaPano/          # ReplicaPano dataset samples
│
├─evaluation/               # Evaluation related code
│  ├─llm_evaluator.py      # LLM evaluator
│  ├─metric.py             # Evaluation metrics
│  └─model_run/           # Model running scripts
│      ├─api_model_run.py 
│      └─open_model_run.py
│
├─result/                  # Evaluation results directory
│
├─scripts/                 # Utility scripts
│  ├─frequency_count.py   
│  ├─negative_sampling.py  
│  ├─QApairGenerate.py    
│  ├─analysis.py          
│  └─delete_null.py       
│
└─example/                 # Example data
```

## Dataset Structure

Each scene folder contains the following files:
- cognitive_map.json: Cognitive map data
- *_qa.json: QA pair data
- *_qa_without_cogmap.json: QA data without cognitive map
- *_qa_with_negative.json: QA data with negative sampling
- rgb.png: RGB image
- room_layout.jpg: Room layout image

We convert all json files to csv for dataset hosting strategy

## Usage

1. Datasets are located in the `data/` directory
2. Evaluation code can be found in `evaluation/` directory
3. Utility scripts are in `scripts/` directory
4. Evaluation results will be saved to `result/` directory

### Example Command

To run evaluation using proprietary model (you need to fill your api-key in `api_model_run.py`):
```bash
python evaluation/model_run/api_model_run.py \
    --input_dir ./example \
    --output_dir ./result \
    --model google/gemini-pro-1.5 \
    --json-suffix qa \
    --prompt-type vanilla
```

To run evaluation using a LOCAL open-sourse model:
```bash
python evaluation/model_run/open_model_run.py \
    --input_dir ./example \
    --output_dir ./result/example \
    --model_path models/llava-v1.5-13b \
    --model_type llava \
    --json-suffix qa \
    --prompt-type vanilla \
    --device cuda \
    --gpu_type A40
```

These commands will:
- Use example data from `./example` directory
- Save results to `./result` directory
- Process QA files with suffix 'qa'
- Use vanilla prompt type for evaluation

The first command uses Google's Gemini-Pro 1.5 model.
The second command uses a local LLaVA model with specified path, device, and GPU type.

[License Information]

MIT License

Copyright (c) 2021 Yuan Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

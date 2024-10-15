# FreeVideoLLM

Free Video-LLM: Prompt-guided Visual Perception for Efficient Training-free Video LLM [[arXiv]](https://arxiv.org/abs/2410.10441) [[code]](https://github.com/contrastive/FreeVideoLLM)

by Kai Han, Jianyuan Guo, Yehui Tang, Wei He, Enhua Wu, Yunhe Wang

## Getting Started

### Installation

- The code is developed with CUDA 11.7, Python >= 3.10.12, PyTorch >= 2.1.0

    1. Install the requirements.
        ```
        bash setup_env.sh
        ```

    2. Add OpenAI key and organization to the system environment to use GPT-3.5-turbo for model evaluation.
        ```
        export OPENAI_API_KEY=$YOUR_OPENAI_API_KEY
        export OPENAI_ORG=$YOUR_OPENAI_ORG  # optional
        ```

    3. Download pre-trained LLaVA-v1.6 weights from [`HuggingFace`](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2), and put them under the [`FreeVideoLLM`](./) folder.
        ```
        git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b liuhaotian/llava-v1.6-vicuna-7b
        git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-34b liuhaotian/llava-v1.6-34b
        ```

### Data Preparation

1. We prepare the ground-truth question and answer files based on [`IG-VLM`](https://github.com/imagegridworth/IG-VLM/tree/main), and put them under [playground/gt_qa_files](playground/gt_qa_files).

    - MSVD-QA
        - Download the `MSVD_QA.csv` from the [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/open_ended_qa/MSVD_QA.csv)
        - Reformat the files by running
            ```
            python scripts/data/prepare_msvd_qa_file.py --qa_file $PATH_TO_CSV_FILE
            ```
    - MSRVTT-QA
        - Download the `MSRVTT_QA.csv` from the [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/open_ended_qa/MSRVTT_QA.csv)
        - Reformat the files by running
            ```
            python scripts/data/prepare_msrvtt_qa_file.py --qa_file $PATH_TO_CSV_FILE
            ```
    - TGIF-QA
        - Download the `TGIF_FrameQA.csv` from the [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/open_ended_qa/TGIF_FrameQA.csv)
        - Reformat the files by running
            ```
            python scripts/data/prepare_tgif_qa_file.py --qa_file $PATH_TO_CSV_FILE
            ```
    - Activitynet-QA
        - Download the `Activitynet_QA.csv` from the [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/open_ended_qa/ActivityNet_QA.csv)
        - Reformat the files by running
            ```
            python scripts/data/prepare_activitynet_qa_file.py --qa_file $PATH_TO_CSV_FILE
            ```

2. Download the raw videos from the official websites.

    - Openset VideoQA

        - [Recomanded] Option 1: Follow the instruction in [`Video-LLaVA`](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md) to download raw videos.
        - Option 2: Download videos from the data owners.
            - [`MSVD-QA`](https://github.com/xudejing/video-question-answering?tab=readme-ov-file)
            - [`MSRVTT-QA`](https://github.com/xudejing/video-question-answering?tab=readme-ov-file)
            - [`TGIF-QA`](https://github.com/YunseokJANG/tgif-qa?tab=readme-ov-file)
            - [`ActivityNet-QA`](https://github.com/MILVLG/activitynet-qa)


3. Organize the raw videos under [playground/data](playground/data).

    - To directly use our data loaders without changing paths, please organize your datasets as follows

        ```
        $ FreeVideoLLM/playground/data
            ├── video_qa
                ├── MSVD_Zero_Shot_QA
                    ├── videos
                        ├── ...
                ├── MSRVTT_Zero_Shot_QA
                    ├── videos
                        ├── all
                            ├── ...
                ├── TGIF_Zero_Shot_QA
                   ├── mp4
                       ├── ...
                ├── Activitynet_Zero_Shot_QA
                   ├── all_test
                       ├── ...
        ```

## Configuration

We use yaml config to control the design choice. You can refer to the code https://github.com/contrastive/FreeVideoLLM/blob/e973c8840306f60773b0d9058b222287c45c5f97/free_video_llm/llava/model/llava_arch.py#L275 to understand the config.

## Inference and Evaluation

FreeVideoLLM is a training-free method, so we can directly do the inference and evaluation without model training.

By default, we use 8 GPUs for the model inference. We can modify the `CUDA_VISIBLE_DEVICES` in the config file to accommodate your own settings. Please note that the model inference of FreeVideoLLM-34B requires GPUs with at least 80G memory.

```
cd FreeVideoLLM
python run_inference.py --exp_config $PATH_TO_CONFIG_FILE
```

- This is optional, but use `export PYTHONWARNINGS="ignore"` if you want to suppress the warnings.

### Output Structures

- The inference outputs will be stored under [`outputs/artifacts`](outputs/artifacts).
- The intermediate outputs of GPT-3.5-turbo will be stored under [`outputs/eval_save_dir`](outputs/eval_save_dir).
- The evaluation results will be stored under [`outputs/logs`](outputs/logs).
- All of these can be changed in the config file.

## Acknowledgement

The project is developed based on [LLaVA-v1.6](https://github.com/haotian-liu/LLaVA), [SlowFast-LLaVA](https://github.com/apple/ml-slowfast-llava), [IG-VLM](https://github.com/imagegridworth/IG-VLM), [CLIP](https://github.com/openai/CLIP) and [transformers](https://github.com/huggingface/transformers).

## Citation
```
@misc{han2024freevideollmpromptguidedvisual,
      title={Free Video-LLM: Prompt-guided Visual Perception for Efficient Training-free Video LLMs}, 
      author={Kai Han and Jianyuan Guo and Yehui Tang and Wei He and Enhua Wu and Yunhe Wang},
      year={2024},
      eprint={2410.10441},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.10441}, 
}
```

# RL

## installation

We suggest you use a different rl environment.

```bash
conda create -n m1_rl python=3.10
conda activate m1_rl
cd verl/
pip install -e .
pip install -r deepscaler-requirements.txt
pip install causal-conv1d>=1.4.0
pip install mamba-ssm
```

This is for evaluation. The training is slow because hybrid models lack features such as data packing in training and batch generation with left-padding tokens for different problems. We are working on supporting those features.

Download datasets

For AIME2025 datasets, please refer to [this](https://github.com/cmu-l3/l1/blob/294b0b19bdd8aa1938ea4c5d7b1a408afbcf9408/scripts/data/generate_aime.py) and only use `num_tokens=-1`.

For other evaluation datastes, please refer to [this](https://github.com/agentica-project/rllm/blob/7d4f1e23729cd6d16eee8457dfbf2f8bc5acdf03/scripts/data/deepscaler_dataset.py).

You can find our dataset [here](https://drive.google.com/drive/folders/1OgkktGEnMb_KIU_BH_7a4A3zYW42GmIW?usp=sharing).

RL model is [here](https://huggingface.co/JunxiongWang/M1-3B). Make sure you save your data under `rl/verl/data/`

```
cd verl/
bash scripts/eval.sh JunxiongWang/M1-3B aime2025 32 1 0.7 32768
bash scripts/eval.sh JunxiongWang/M1-3B aime 32 1 0.7 32768
bash scripts/eval.sh JunxiongWang/M1-3B math 32 1 0.7 32768
bash scripts/eval.sh JunxiongWang/M1-3B olympiad_bench  32 1 0.7 32768
```

Most of code is copied from [here](https://github.com/volcengine/verl)


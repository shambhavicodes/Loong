# M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models

M1 presents a distilled hybrid architecture that achieves efficient and effective reasoning capabilities.

This repository includes everything you need to **distill and create strong R1-style reasoning models** using the **hybrid Mamba architecture**. It serves as both a toolkit and a reference implementation for building hybrid mamba reasoning models.

### Environment Setup
Create a new Conda environment and install the necessary dependencies:

```
conda create -n m1 python=3.10
conda activate m1
pip install torch==2.4.0
pip install causal-conv1d>=1.4.0
pip install mamba-ssm
```

### Inference

```python
import torch
from transformers import AutoTokenizer
from mamba.hybrid_wrapper import MambaTransformerHybridModelWrapper

pretrained_model_name = "togethercomputer/M1-3B"
model = MambaTransformerHybridModelWrapper.from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16)
model.eval()

messages = [[
    {
        "role": "user",
        "content": "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$",
    },
]]

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
formatted_prompts = [
    tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages
]

prompts = [
    tokenizer.encode(formatted_prompt, return_tensors="pt", truncation=True, max_length=200)
    for formatted_prompt in formatted_prompts
]
batch_prompts = torch.cat(prompts, dim=0).cuda()

outputs = model.generate(
    input_ids=batch_prompts,
    max_length=8000,
    cg=True,
    return_dict_in_generate=True,
    output_scores=True,
    enable_timing=True,
    top_k=1,
    temperature=0.7,
    eos_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.batch_decode(outputs.sequences.tolist())
print(generated_text[0])
```

### Evaluation

Please refer to [here](rl/README.md)

| **Model**                          | **AIME 2025** | **AIME 2024** | **MATH 500** | **AMC 2023** | **OlympiadBench** |
|-----------------------------------|---------------|---------------|--------------|--------------|-------------------|
| Qwen2.5-Math-7B-Instruct  (Transformer)        | –             | 13.3          | 79.8         | 50.6         | 40.7              |
| rStar-Math-7B  (Transformer)                   | –             | 26.7          | 78.4         | 47.5         | 47.1              |
| Eurus-2-7B-PRIME (Transformer)                 | –             | 26.7          | 79.2         | 57.8         | 42.1              |
| Qwen2.5-7B-SimpleRL (Transformer)              | –             | 26.7          | 82.4         | 62.5         | 43.3              |
| DeepSeek-R1-Distill-Qwen-1.5B (Transformer)    | 23.0          | 28.8          | 82.8         | 62.9         | 43.3              |
| [**M1-3B (Mamba Hybrid Models)**](https://huggingface.co/togethercomputer/M1-3B)                | 23.5          | 28.5          | 84.0         | 62.8         | 47.3              |

### Training

* Initialization. Please update the model name according to your specifications. This repository supports any model (e.g., Llama, Qwen, R1 Distilled Qwen) available on Hugging Face.

  `python transformer_to_mamba.py`

* Distillation + SFT. For demonstration, we use [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) as the distillation dataset, which was generated from the LlaMA 405B model. If you want to perform distillation from Qwen or other models, please consider using a dataset generated from those models instead.

  Please refer to [here](sft/README.md)

* RL. We provide an RL training framework built on top of [VeRL](https://github.com/volcengine/verl).

  Please refer to [here](rl/README.md)

Most training frameworks, both SFT and RL, require data packing features with `position_ids` (or `cu_seqlens` in FlashAttention) to prevent data contamination. In our experience, using `position_ids` to prevent data contamination in SFT might not be necessary. However, to train hybrid models with RL, you must install [Packed Mamba Hybrid](HYBRID_PACK.md) and use `position_ids` to avoid the data contamination. Packing data in the same sequence without using `position_ids` will cause the RL training to fail.

### Citation

If you use this codebase or find our work valuable, please consider citing:

```
@article{wang2025m1scalabletesttimecompute,
  title={M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models}, 
  author={Junxiong Wang and Wen-Ding Li and Daniele Paliotta and Daniel Ritter and Alexander M. Rush and Tri Dao},
  journal={arXiv preprint arXiv:2504.10449},
  year={2025},
  url={https://arxiv.org/abs/2504.10449}, 
}

@article{paliotta2025thinking,
  title={Thinking slow, fast: Scaling inference compute with distilled reasoners},
  author={Paliotta, Daniele and Wang, Junxiong and Pagliardini, Matteo and Li, Kevin Y and Bick, Aviv and Kolter, J Zico and Gu, Albert and Fleuret, Fran{\c{c}}ois and Dao, Tri},
  journal={arXiv preprint arXiv:2502.20339},
  year={2025}
}

@inproceedings{junxiongdaniele2024mambainllama,
  title={The Mamba in the Llama: Distilling and Accelerating Hybrid Models},
  author={Junxiong Wang and Daniele Paliotta and Avner May and Alexander M Rush and Tri Dao},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=uAzhODjALU}
}
```

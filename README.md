# M1

### Environment Setup
Create a new Conda environment and install the necessary dependencies:

```
conda create -n m1 python=3.10
conda activate m1
pip install torch==2.4.0
pip install causal-conv1d>=1.4.0
pip install mamba-ssm
pip install transformers==4.47.1
```

### Inference

```
import torch
from transformers import AutoTokenizer
from mamba.hybrid_wrapper import MambaTransformerHybridModelWrapper

pretrained_model_name = "JunxiongWang/M1-3B"
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

### Traning

* Initialization. Please update the model name according to your specification.

  `python transformer_to_mamba.py`

* Distillation + SFT

  Please refer to [here](sft/README.md)

* RL

  Please refer to [here](rl/README.md)

### Evalution

Please refer to [here](rl/README.md)

### Citation

If you use this codebase, or otherwise found our work valuable, please cite:

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
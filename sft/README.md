**Requirements**: Python >=3.10 and Pytorch >=2.1.1.

```bash
pip install flash-attn==2.7.2.post1
pip install deepspeed==0.14.4
pip install -e .
```

### Usage

Distillation over OpenMathInstruct

```bash
accelerate launch -m axolotl.cli.train math_config/distill.yaml
```

SFT over OpenMathInstruct

```bash
accelerate launch -m axolotl.cli.train math_config/sft.yaml
```

SFT over Reasoning

```bash
accelerate launch -m axolotl.cli.train reason_config/sft.yaml
```

If you want to build your dataset, please refer `tokenized_dataset.py`

Most of code is copied from [here](https://github.com/axolotl-ai-cloud/axolotl)
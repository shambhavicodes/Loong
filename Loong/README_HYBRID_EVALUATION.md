# Hybrid Model Loong Evaluation

This directory contains the setup for evaluating hybrid Mamba-Transformer models using the Loong benchmark.

## Overview

The evaluation compares:
- **Hybrid Distilled Model**: Our distilled hybrid Mamba-Transformer model
- **Teacher Model**: Llama-3.2-3B-Instruct (the original teacher model)
- **Judge Model**: GPT-4 Turbo (recommended) or Qwen2.5-7B-Instruct (for evaluation scoring)

## Quick Start

### Option A: Using Local Models (vLLM)

#### 1. Start the Models

```bash
# Start all required vLLM servers
./start_models.sh
```

This will start:
- Hybrid model on port 8000
- Qwen judge on port 8001  
- Teacher model on port 8002

Wait 2-3 minutes for models to fully load.

#### 2. Run Quick Test

```bash
# Test with 3 samples to verify everything works
./run_quick_test.sh
```

#### 3. Run Full Evaluation

```bash
# Run complete evaluation pipeline
./run_hybrid_evaluation.sh
```

### Option B: Using GPT-4 as Judge (Recommended)

#### 1. Start Models (Hybrid + Teacher only)

```bash
# Start hybrid and teacher models only
./start_models.sh
```

#### 2. Run Quick Test with GPT-4

```bash
# Test with 3 samples using GPT-4 as judge
./run_quick_test_gpt4.sh
```

#### 3. Run Full Evaluation with GPT-4

```bash
# Run complete evaluation pipeline with GPT-4 judge
./run_hybrid_evaluation_gpt4.sh
```

## Configuration Files

- `config/models/hybrid_distilled.yaml` - Hybrid model configuration
- `config/models/teacher_model.yaml` - Teacher model configuration  
- `config/models/qwen_judge.yaml` - Judge model configuration

## Output

Results are saved in:
- `output/hybrid_evaluation/` - Full evaluation results
- `output/quick_test/` - Quick test results
- `logs/` - Server logs

## Requirements

- Conda environments: `verl_eval` and `loong`
- vLLM for model serving
- Sufficient GPU memory (recommend 24GB+ per model)

## Troubleshooting

1. **Port conflicts**: Ensure ports 8000, 8001, 8002 are available
2. **GPU memory**: Reduce `--gpu-memory-utilization` if OOM errors occur
3. **Model loading**: Check `logs/` directory for server startup issues

## Evaluation Pipeline

1. **Step 1**: Generate responses with hybrid model
2. **Step 2**: Generate responses with teacher model  
3. **Step 3**: Evaluate hybrid model responses
4. **Step 4**: Evaluate teacher model responses
5. **Step 5**: Calculate metrics and compare performance

The evaluation uses the standard Loong benchmark format and scoring methodology.

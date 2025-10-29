#!/bin/bash

# Start vLLM servers for hybrid model evaluation
# This script starts the required models on different ports

echo "Starting vLLM servers for hybrid model evaluation..."

# Start Hybrid Distilled Model (port 8000)
echo "Starting Hybrid Distilled Model on port 8000..."
nohup bash -c "eval \"\$(conda shell.bash hook)\" && conda activate verl_eval && vllm serve shivi101/tulu3-hybrid-ssm50-50 --port 8000 --host 0.0.0.0 --tensor-parallel-size 1 --gpu-memory-utilization 0.6 --trust-remote-code" > logs/hybrid_model.log 2>&1 &

# Start Qwen Judge Model (port 8001)
echo "Starting Qwen Judge Model on port 8001..."
nohup bash -c "eval \"\$(conda shell.bash hook)\" && conda activate loong && vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 --host 0.0.0.0 --tensor-parallel-size 1 --gpu-memory-utilization 0.5" > logs/qwen_judge.log 2>&1 &

# Start Teacher Model (port 8002)
echo "Starting Teacher Model on port 8002..."
nohup bash -c "eval \"\$(conda shell.bash hook)\" && conda activate loong && vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8002 --host 0.0.0.0 --tensor-parallel-size 1 --gpu-memory-utilization 0.5" > logs/teacher_model.log 2>&1 &

echo "All models starting in background..."
echo "Check logs/ directory for server logs"
echo "Wait 2-3 minutes for models to fully load before running evaluation"

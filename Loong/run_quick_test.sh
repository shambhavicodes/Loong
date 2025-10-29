#!/bin/bash

# Quick test script for hybrid model evaluation
# Tests with just 3 samples to verify everything works

set -e

echo "=== Quick Test: Hybrid Model Evaluation ==="
echo "Testing with 3 samples..."

# Configuration
MODEL_NAME="hybrid_distilled"
JUDGE_MODEL="qwen_judge"
OUTPUT_DIR="output/quick_test"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Generate responses with hybrid model (3 samples)
echo "=== Step 1: Generate responses (3 samples) ==="
python src/step2_model_generate.py \
    --models "$MODEL_NAME" \
    --model_config_dir config/models \
    --input_path data/loong.jsonl \
    --output_path "$OUTPUT_DIR/hybrid_generate.jsonl" \
    --process_num_gen 1 \
    --debug_num 3

# Step 2: Evaluate responses
echo "=== Step 2: Evaluate responses ==="
python src/step3_model_evaluate.py \
    --model "$MODEL_NAME" \
    --eval_model "$JUDGE_MODEL" \
    --model_config_dir config/models \
    --input_path "$OUTPUT_DIR/hybrid_generate.jsonl" \
    --output_path "$OUTPUT_DIR/hybrid_generate.jsonl" \
    --evaluate_output_path "$OUTPUT_DIR/hybrid_evaluate.jsonl" \
    --process_num_eval 1 \
    --debug_num 3

# Step 3: Calculate metrics
echo "=== Step 3: Calculate metrics ==="
python src/step4_cal_metric.py \
    --input_path "$OUTPUT_DIR/hybrid_evaluate.jsonl" \
    --output_path "$OUTPUT_DIR/hybrid_metrics.json"

# Step 4: Display results
echo "=== Results ==="
cat "$OUTPUT_DIR/hybrid_metrics.json"

echo ""
echo "=== Quick Test Complete ==="
echo "If this works, you can run the full evaluation with: ./run_hybrid_evaluation.sh"

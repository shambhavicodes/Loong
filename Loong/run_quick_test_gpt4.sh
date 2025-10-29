#!/bin/bash

# Quick test script for hybrid model evaluation using GPT-4 as judge
# Tests with just 3 samples to verify everything works

set -e

echo "=== Quick Test: Hybrid Model Evaluation with GPT-4 Judge ==="
echo "Testing with 3 samples..."

# Configuration
MODEL_NAME="hybrid_distilled"
JUDGE_MODEL="gpt4"  # Using GPT-4 as judge instead of Qwen
OUTPUT_DIR="output/quick_test_gpt4"

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

# Step 2: Evaluate responses using GPT-4
echo "=== Step 2: Evaluate responses with GPT-4 ==="
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
echo "GPT-4 evaluation results saved in: $OUTPUT_DIR/"
echo "If this works, you can run the full evaluation with GPT-4 judge"

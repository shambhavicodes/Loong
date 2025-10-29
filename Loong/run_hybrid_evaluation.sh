#!/bin/bash

# Run Loong evaluation for hybrid distilled model
# This script runs the complete evaluation pipeline

set -e

# Configuration
MODEL_NAME="hybrid_distilled"
TEACHER_MODEL="teacher_model"
JUDGE_MODEL="qwen_judge"
OUTPUT_DIR="output/hybrid_evaluation"
DEBUG_NUM=10  # Set to -1 for full evaluation

echo "=== Hybrid Model Loong Evaluation ==="
echo "Model: $MODEL_NAME"
echo "Teacher: $TEACHER_MODEL"
echo "Judge: $JUDGE_MODEL"
echo "Output: $OUTPUT_DIR"
echo "Debug samples: $DEBUG_NUM"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Step 1: Generate responses with hybrid model
echo "=== Step 1: Generate responses with Hybrid Model ==="
python src/step2_model_generate.py \
    --models "$MODEL_NAME" \
    --model_config_dir config/models \
    --input_path data/loong.jsonl \
    --output_path "$OUTPUT_DIR/hybrid_generate.jsonl" \
    --process_num_gen 5 \
    --debug_num $DEBUG_NUM

# Step 2: Generate responses with teacher model (for comparison)
echo "=== Step 2: Generate responses with Teacher Model ==="
python src/step2_model_generate.py \
    --models "$TEACHER_MODEL" \
    --model_config_dir config/models \
    --input_path data/loong.jsonl \
    --output_path "$OUTPUT_DIR/teacher_generate.jsonl" \
    --process_num_gen 5 \
    --debug_num $DEBUG_NUM

# Step 3: Evaluate hybrid model responses
echo "=== Step 3: Evaluate Hybrid Model responses ==="
python src/step3_model_evaluate.py \
    --model "$MODEL_NAME" \
    --eval_model "$JUDGE_MODEL" \
    --model_config_dir config/models \
    --input_path "$OUTPUT_DIR/hybrid_generate.jsonl" \
    --output_path "$OUTPUT_DIR/hybrid_generate.jsonl" \
    --evaluate_output_path "$OUTPUT_DIR/hybrid_evaluate.jsonl" \
    --process_num_eval 5 \
    --debug_num $DEBUG_NUM

# Step 4: Evaluate teacher model responses
echo "=== Step 4: Evaluate Teacher Model responses ==="
python src/step3_model_evaluate.py \
    --model "$TEACHER_MODEL" \
    --eval_model "$JUDGE_MODEL" \
    --model_config_dir config/models \
    --input_path "$OUTPUT_DIR/teacher_generate.jsonl" \
    --output_path "$OUTPUT_DIR/teacher_generate.jsonl" \
    --evaluate_output_path "$OUTPUT_DIR/teacher_evaluate.jsonl" \
    --process_num_eval 5 \
    --debug_num $DEBUG_NUM

# Step 5: Calculate metrics
echo "=== Step 5: Calculate metrics ==="
python src/step4_cal_metric.py \
    --input_path "$OUTPUT_DIR/hybrid_evaluate.jsonl" \
    --output_path "$OUTPUT_DIR/hybrid_metrics.json"

python src/step4_cal_metric.py \
    --input_path "$OUTPUT_DIR/teacher_evaluate.jsonl" \
    --output_path "$OUTPUT_DIR/teacher_metrics.json"

# Step 6: Display results
echo "=== Step 6: Results ==="
echo "Hybrid Model Metrics:"
cat "$OUTPUT_DIR/hybrid_metrics.json"
echo ""
echo "Teacher Model Metrics:"
cat "$OUTPUT_DIR/teacher_metrics.json"

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved in: $OUTPUT_DIR/"

#!/bin/bash

# Comprehensive evaluation script for all hybrid models
# This script runs evaluation for multiple models and monitors progress

MODELS=(
    "hybrid_ssm25-75"
    "hybrid_ssm50-50" 
    "hybrid_ssm75-25"
    "control_ssm25-75"
    "control_ssm50-50"
    "control_ssm75-25"
    "llama3.2-3b"
)

MODEL_PATHS=(
    "/home/toolkit/M1/outputs/tulu3_distill_hybrid_ssm25-75_8gpu"
    "/home/toolkit/M1/outputs/tulu3_distill_hybrid_ssm50-50_8gpu"
    "/home/toolkit/M1/outputs/tulu3_distill_hybrid_ssm75-25_8gpu"
    "/home/toolkit/M1/outputs/tulu3_distill_control_ssm25-75_8gpu"
    "/home/toolkit/M1/outputs/tulu3_distill_control_ssm50-50_8gpu"
    "/home/toolkit/M1/outputs/tulu3_distill_control_ssm75-25_8gpu"
    "meta-llama/Llama-3.2-3B-Instruct"
)

echo "Starting comprehensive evaluation of ${#MODELS[@]} models..."
echo "Models: ${MODELS[*]}"
echo "Start time: $(date)"

# Create results directory
mkdir -p results
RESULTS_FILE="results/evaluation_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "Evaluation Summary - $(date)" > $RESULTS_FILE
echo "=================================" >> $RESULTS_FILE

# Function to run evaluation for a single model
run_model_evaluation() {
    local model_name=$1
    local model_path=$2
    local start_time=$(date +%s)
    
    echo "Starting evaluation for $model_name at $(date)"
    echo "Model path: $model_path"
    
    # Create model config if it doesn't exist
    config_file="config/models/${model_name}.yaml"
    if [ ! -f "$config_file" ]; then
        echo "Creating config for $model_name"
        cat > "$config_file" << EOF
type: "hybrid"
args:
  model_path: "$model_path"
run_args:
  temperature: 0.0
  top_p: 1.0
EOF
    fi
    
    # Run evaluation
    cd src
    echo "Running evaluation for $model_name..."
    
    # Use screen to run in background
    screen -dmS "eval_${model_name}" bash -c "
        echo 'Starting evaluation for $model_name at \$(date)'
        python step1_load_data.py --models ${model_name}.yaml --eval_model gpt4.yaml --debug_num -1 --doc_path ../data/doc --input_path ../data/loong.jsonl --output_process_path ../data/loong_process_${model_name}.jsonl --output_path ../output/$model_name/loong_generate.jsonl --evaluate_output_path ../output/$model_name/loong_evaluate.jsonl --max_length 128000 --model_config_dir ../config/models --process_num_gen 1 --process_num_eval 20
        
        python step2_model_generate.py --models ${model_name}.yaml --eval_model gpt4.yaml --debug_num -1 --doc_path ../data/doc --input_path ../data/loong.jsonl --output_process_path ../data/loong_process_${model_name}.jsonl --output_path ../output/$model_name/loong_generate.jsonl --evaluate_output_path ../output/$model_name/loong_evaluate.jsonl --max_length 128000 --model_config_dir ../config/models --process_num_gen 1 --process_num_eval 20
        
        python step3_model_evaluate.py --models ${model_name}.yaml --eval_model gpt4.yaml --debug_num -1 --doc_path ../data/doc --input_path ../data/loong.jsonl --output_process_path ../data/loong_process_${model_name}.jsonl --output_path ../output/$model_name/loong_generate.jsonl --evaluate_output_path ../output/$model_name/loong_evaluate.jsonl --max_length 128000 --model_config_dir ../config/models --process_num_gen 1 --process_num_eval 20
        
        python step4_cal_metric.py --models ${model_name}.yaml --eval_model gpt4.yaml --debug_num -1 --doc_path ../data/doc --input_path ../data/loong.jsonl --output_process_path ../data/loong_process_${model_name}.jsonl --output_path ../output/$model_name/loong_generate.jsonl --evaluate_output_path ../output/$model_name/loong_evaluate.jsonl --max_length 128000 --model_config_dir ../config/models --process_num_gen 1 --process_num_eval 20 > ../results/${model_name}_metrics.txt
        
        echo 'Completed evaluation for $model_name at \$(date)'
    "
    
    cd ..
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "Started evaluation for $model_name (Duration: ${duration}s)"
    echo "Started evaluation for $model_name (Duration: ${duration}s)" >> $RESULTS_FILE
}

# Start all evaluations
for i in "${!MODELS[@]}"; do
    run_model_evaluation "${MODELS[$i]}" "${MODEL_PATHS[$i]}"
    sleep 10  # Small delay between starting evaluations
done

echo "All evaluations started!"
echo "Use 'screen -ls' to see running evaluations"
echo "Use 'screen -r eval_<model_name>' to attach to a specific evaluation"
echo ""
echo "Monitor progress with:"
echo "  ./monitor_progress.sh"
echo ""
echo "Results will be saved in: results/"
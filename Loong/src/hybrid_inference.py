"""
Hybrid model inference module for Loong benchmark evaluation.
This module provides integration with MambaTransformerHybridModelWrapper from the verl library.
"""

import torch
from transformers import AutoTokenizer
from verl.trainer import MambaTransformerHybridModelWrapper


def load_hybrid_model(model_path):
    """
    Load a hybrid model and tokenizer from the specified path.
    
    Args:
        model_path (str): Path to the hybrid model
        
    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails
    """
    try:
        print(f"Loading hybrid model from: {model_path}")
        
        # Load the hybrid model using MambaTransformerHybridModelWrapper
        model = MambaTransformerHybridModelWrapper.from_pretrained(
            model_path,
            pretrained_model_name="meta-llama/Llama-3.2-3B-Instruct",
            attn_implementation="flash_attention_2"
        )
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            
        print(f"Successfully loaded hybrid model: {model_path}")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading hybrid model from {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def generate_hybrid_response(prompt_input, config):
    """
    Generate response using hybrid model with proper error handling.
    
    Args:
        prompt_input (dict): Input containing 'prompt' field
        config (dict): Configuration containing model path and generation args
        
    Returns:
        str: Generated response or error message
    """
    prompt = prompt_input['prompt']
    model_path = config['args']['model_path']

    model, tokenizer = load_hybrid_model(model_path)
    if model is None or tokenizer is None:
        return f"ERROR: Failed to load hybrid model from {model_path}"

    try:
        # Tokenize input with proper truncation
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Get generation parameters from config
        temperature = config['run_args'].get('temperature', 0.0)
        max_new_tokens = config['run_args'].get('max_new_tokens', 512)
        top_p = config['run_args'].get('top_p', 1.0)

        # Clear model state before generation to avoid contamination
        if hasattr(model, 'reset_cache'):
            model.reset_cache()
        elif hasattr(model, 'clear_cache'):
            model.clear_cache()

        # Clear CUDA cache to ensure clean state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            try:
                # Generate response with proper parameters
                output = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    do_sample=False,  # Always use greedy decoding to avoid numerical issues
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.1,
                    output_scores=False,
                    return_dict_in_generate=False
                )
            except RuntimeError as e:
                if "probability tensor contains either `inf`, `nan` or element < 0" in str(e):
                    # Try with more conservative settings
                    print(f"Generation failed, trying with minimal settings: {e}")
                    try:
                        output = model.generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            do_sample=False,
                            max_new_tokens=min(max_new_tokens, 256),  # Reduce max tokens
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            temperature=0.0,
                            top_p=1.0,
                            repetition_penalty=1.0,
                            output_scores=False,
                            return_dict_in_generate=False
                        )
                    except RuntimeError as e2:
                        print(f"Even minimal generation failed: {e2}")
                        # Return a placeholder response
                        return "I apologize, but I encountered a technical issue generating a response for this question."
                else:
                    raise e

        # Decode the response
        response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response

    except Exception as e:
        print(f"Error in hybrid generation: {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {e}"

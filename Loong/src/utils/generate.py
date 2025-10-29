import json
from tqdm import tqdm
import multiprocessing
import requests
import numpy as np
from functools import partial
from decimal import Decimal
import numpy as np
import time
import torch
import sys
import os
from pathlib import Path

# Optional imports for different model types
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            try:
                return str(obj, encoding='utf-8')
            except:
                return str(obj, encoding='gbk')
        elif isinstance(obj, Decimal):
            return float(obj)
        # print(obj, type(obj))
        return json.JSONEncoder.default(self, obj)


# Global hybrid model cache to avoid reloading
_hybrid_model_cache = {}

def load_hybrid_model(model_path):
    """Load hybrid model with caching to avoid reloading"""
    if model_path in _hybrid_model_cache:
        return _hybrid_model_cache[model_path]
    
    print(f"Loading hybrid model from: {model_path}")
    
    # Add the path to VERL's hybrid model wrapper
    sys.path.append('/home/toolkit/M1/rl/verl')
    from verl.models.mamba_inference.hybrid_wrapper import MambaTransformerHybridModelWrapper
    from transformers import AutoTokenizer, GenerationConfig
    
    try:
        # Use VERL's hybrid wrapper to properly load the model (following eval.sh pattern)
        model = MambaTransformerHybridModelWrapper.from_pretrained(
            pretrained_model_name=model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"✅ Hybrid model moved to GPU: {next(model.parameters()).device}")
        else:
            print("⚠️ CUDA not available, model will run on CPU (performance may be slow).")
        
        _hybrid_model_cache[model_path] = (model, tokenizer)
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading hybrid model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_hybrid_response(prompt_input, config):
    """Generate response using hybrid model following eval.sh pattern"""
    prompt = prompt_input['prompt']
    model_path = config['args']['model_path']
    
    model, tokenizer = load_hybrid_model(model_path)
    if model is None or tokenizer is None:
        return f"ERROR: Failed to load hybrid model from {model_path}"
    
    try:
        # Tokenize with proper truncation (following eval.sh pattern with prompt_length=1024)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generation config matching eval.sh parameters
        from transformers import GenerationConfig
        temperature = config['run_args'].get('temperature', 0.0)
        max_new_tokens = config['run_args'].get('max_new_tokens', 512)
        top_p = config['run_args'].get('top_p', 1.0)
        
        # Generation config following eval.sh pattern - always use greedy decoding for stability
        generation_config = GenerationConfig(
            do_sample=False,  # Always use greedy decoding to avoid numerical issues
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

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
                # Try with minimal parameters first
                output = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    do_sample=False,
                    max_new_tokens=min(max_new_tokens, 256),  # Reduce max tokens to avoid issues
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    generation_config=generation_config,
                    output_scores=False,
                    return_dict_in_generate=False
                )
            except RuntimeError as e:
                if "probability tensor contains either `inf`, `nan` or element < 0" in str(e):
                    # Try with even more conservative settings
                    print(f"Generation failed, trying with minimal settings: {e}")
                    try:
                        fallback_config = GenerationConfig(
                            do_sample=False,
                            max_new_tokens=50,  # Very conservative
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        output = model.generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            do_sample=False,
                            max_new_tokens=50,  # Very conservative
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            generation_config=fallback_config,
                            output_scores=False,
                            return_dict_in_generate=False
                        )
                    except RuntimeError as e2:
                        print(f"Even minimal generation failed: {e2}")
                        # Return a placeholder response
                        return "I apologize, but I encountered a technical issue generating a response for this question."
                else:
                    raise e
        
        response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response
        
    except Exception as e:
        print(f"Error in hybrid generation: {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {e}"

def get_api_results(prompt_input, config):
    prompt = prompt_input['prompt']

    if config['type'] == 'hybrid':
        return generate_hybrid_response(prompt_input, config)
    elif config['type'] == 'openai' or config['type'] == 'vllm':
        if OpenAI is None:
            return "ERROR: OpenAI library not installed"
        client = OpenAI(api_key=config['args']['api_key'],
                        base_url=config['args']['api_url'] if config['args']['api_url']!='' else None)
        try: 
            response = client.chat.completions.create(
                messages=[{"role": "user","content": prompt}],
                model=config['args']['api_name'],
                temperature=config['run_args']['temperature']
            )
            return response.choices[0].message.content
        except Exception as e:
                print(e)
                return []
        
    elif config['type'] == 'gemini':
        if genai is None:
            return "ERROR: Google Generative AI library not installed"
        genai.configure(api_key=config['args']['api_key'])

        model = genai.GenerativeModel(name=config['args']['api_name'])
        try:
            response = model.generate_content(prompt,
                        generation_config=genai.types.GenerationConfig(
                        temperature=config['run_args']['temperature']))
            return response.text
        except Exception as e:
            print(e)
            return []
    
    elif config['type'] == 'claude':
        if Anthropic is None:
            return "ERROR: Anthropic library not installed"
        client = Anthropic(api_key=config['args']['api_key'])
        try:
            message = client.messages.create(
                messages=[{"role": "user", "content": prompt,}],
                model=config['args']['api_name'],
            )
            return message.content
        except Exception as e:
            print(e)
            return []
    
    elif config['type'] == 'http':
        headers = {"Content-Type": "application/json",
                "Authorization": config['args']['api_key']}
        raw_info = {
            "model": config['args']['api_name'],
            "messages": [{"role": "user", "content": prompt}],
            "n": 1}
        raw_info.update(config['run_args'])
        try:
            callback = requests.post(config['args']['api_url'], data=json.dumps(raw_info, cls=MyEncoder), headers=headers,
                                    timeout=(600, 600))
            result = callback.json()
            # todo: customize the result
            return result['data']['response']['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            return []
        
    else:
        raise f"type of {config['type']} is not valid"

def fetch_api_result(prompt_input, config, max_retries=5):
    """Attempt to get a valid result from the API, with a maximum number of retries."""
    for _ in range(max_retries):
        result = get_api_results(prompt_input, config)
        if result: 
            return result
        # Sleep briefly to not hammer the API in case of errors or rate limits
        time.sleep(5) # Uncomment if needed
    return None


def api(prompt, output_path, config, tag):
    response_content = fetch_api_result(prompt, config)
    result = prompt.copy()
    result[tag] = response_content or ""
    with open(output_path, 'a', encoding='utf-8') as fw:
        fw.write(json.dumps(result, ensure_ascii=False) + '\n')


def generate(prompts, config, output_path, process_num, tag):
    func = partial(api, output_path=output_path, config=config, tag=tag)
    with multiprocessing.Pool(processes=process_num) as pool:
        for _ in tqdm(pool.imap(func, prompts), total=len(prompts)):
            pass
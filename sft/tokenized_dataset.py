from transformers import AutoTokenizer
from datasets import load_dataset

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

import torch
import os
import pandas as pd

from tqdm import tqdm

dataset = load_dataset("open-r1/OpenThoughts-114k-math")['train']
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

start_partern = torch.tensor([78191, 128007, 271])
end_partern = torch.tensor([tokenizer.eos_token_id])

conversations_output = []
for dataitem in tqdm(dataset):
    conversations = dataitem["conversations"]
    conversation_items = []
    if len(conversations) != 2:
        print(conversations)
    for conversation in conversations:
        if conversation["from"] == "user":
            conversation_items.append({"role": "user", "content": conversation["value"]})
        elif conversation["from"] == "assistant":
            conversation_items.append({"role": "assistant", "content": conversation["value"]})
    conversations_output.append(conversation_items)

print(len(conversations_output))

# Function to mask patterns
def mask_pattern(tensor, start_pattern, end_pattern, IGNORE_INDEX):
    mask = torch.full(tensor.shape, IGNORE_INDEX)
    start_pattern_length = len(start_pattern)
    end_pattern_length = len(end_pattern)

    i = 0
    in_sequence = False
    start_index = 0

    while i < len(tensor):
        if not in_sequence and i <= len(tensor) - start_pattern_length and torch.equal(tensor[i:i+start_pattern_length], start_pattern):
            in_sequence = True
            start_index = i + start_pattern_length
            i += start_pattern_length - 1
        elif in_sequence and i <= len(tensor) - end_pattern_length and torch.equal(tensor[i:i+end_pattern_length], end_pattern):
            mask[start_index:i+1] = tensor[start_index:i+1]
            in_sequence = False
            i += end_pattern_length - 1
        i += 1

    if in_sequence:
        mask[start_index:] = tensor[start_index:]
    return mask

# Function to process a single line
def process_line(conversations):
    templated_convs = tokenizer.apply_chat_template(conversations, tokenize=False)
    # print(templated_convs)
    tokens = tokenizer(templated_convs, return_tensors="pt")
    input_ids = tokens['input_ids'].squeeze(dim=0)
    attention_mask = tokens['attention_mask'].squeeze(dim=0)
    labels = mask_pattern(input_ids, start_partern, end_partern, -100)
    return {"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist(), "labels": labels.tolist()}

# Read input file
output_file = "/data/junxiong/OpenThoughts2M_llama_math/tokenized_r1/"
total_lines = len(conversations_output)

# Parallel processing
num_workers = 64
num_parts = 128
tokenized_output = []

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = executor.map(process_line, conversations_output, chunksize=100)
    for result in tqdm(futures, total=total_lines, desc="Processing Data", dynamic_ncols=True):
        tokenized_output.append(result)

def write_parquet_chunk(output_chunk, output_dir, part, total_parts):
    """
    Writes a chunk of data to a Parquet file in the format: train-m-of-n.parquet.

    :param output_chunk: List of dictionaries (chunk of data).
    :param output_dir: Directory to save Parquet files.
    :param part: Current part number (m).
    :param total_parts: Total number of parts (n).
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'train-{part}-of-{total_parts}.parquet')
    
    df = pd.DataFrame(output_chunk)
    df.to_parquet(output_file, index=False)
    
    print(f"Saved {output_file}")

def split_and_write_parallel(output, output_dir, num_parts=4, max_workers=4):
    """
    Splits the data and writes it to Parquet files in parallel.

    :param output: Complete list of dictionaries to split.
    :param output_dir: Directory to save Parquet files.
    :param num_parts: Number of splits/parts.
    :param max_workers: Number of parallel threads.
    """
    # Split data into chunks
    chunk_size = len(output) // num_parts
    chunks = [output[i * chunk_size:(i + 1) * chunk_size] for i in range(num_parts)]
    
    # Ensure all data is included in the last chunk
    if len(output) % num_parts != 0:
        chunks[-1].extend(output[num_parts * chunk_size:])
    
    # Write chunks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            part = i + 1
            futures.append(executor.submit(write_parquet_chunk, chunk, output_dir, part, num_parts))
        
        # Display progress
        for future in tqdm(futures, desc="Writing Data in Parallel"):
            future.result()  # Wait for each thread to finish

split_and_write_parallel(tokenized_output, output_dir=output_file, num_parts=num_parts, max_workers=num_workers)

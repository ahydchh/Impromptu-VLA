
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference script for MiniCPM-V-2_6.")
    parser.add_argument("--save_name", type=str, required=True, help="Name of the output file to save predictions.")
    parser.add_argument("--temperature", type=float, default=0.01, help="Temperature for model generation.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=384, help="Number of workers for image loading")
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--input_path", type=str,default="/share/gha/workspace/DriveEMMA/nuScenes2k.json")
    return parser.parse_args()

args = parse_args()
path = args.input_path
MODEL_NAME =args.model_name_or_path  


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=1,
    trust_remote_code=True,
    max_model_len=5120,
    tensor_parallel_size=1  
)

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def process_batch(batch_queries):

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_idx = {
            executor.submit(load_image, q['images'][0]): idx
            for idx, q in enumerate(batch_queries)
        }
        images = [None] * len(batch_queries)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            images[idx] = future.result()


    prompts = []
    for q in batch_queries:
        question = q['messages'][0]['content'].replace(": (1) <image>", "(<image>./</image>)")
        messages = [{'role': 'user', 'content': question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)


    requests = [{
        "prompt": prompt,
        "multi_modal_data": {"image": image}
    } for prompt, image in zip(prompts, images)]

    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    
    sampling_params = SamplingParams(
        stop_token_ids=stop_token_ids,
        temperature=args.temperature,
        max_tokens=4096
    )


    outputs = llm.generate(requests, sampling_params=sampling_params)
    

    return [
        (q["images"], output.outputs[0].text)
        for q, output in zip(batch_queries, outputs)
    ]


def main():

    with open(path, "r") as f:
        data = json.load(f)
    

    save_dir = os.path.dirname(args.save_name)
    os.makedirs(save_dir, exist_ok=True)
    

    if os.path.exists(args.save_name):
        os.remove(args.save_name)
    

    total = len(data)
    progress_bar = tqdm(total=total, desc="Processing")
    
    for i in range(0, len(data), args.batch_size):
        batch = data[i:i+args.batch_size]
        batch_results = process_batch(batch)
        

        with open(args.save_name, "a") as f:
            for images, pred in batch_results:
                f.write(json.dumps({"images": images, "predict": pred}) + "\n")
        
        progress_bar.update(len(batch))
    
    progress_bar.close()

if __name__ == "__main__":
    main()
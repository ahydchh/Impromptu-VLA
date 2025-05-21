import json
import os
import argparse
from tqdm import tqdm
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
import torch
def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for InternVL2_5-8B")
    parser.add_argument("--save_name", type=str, required=True, help="Output file path")
    parser.add_argument("--temperature", type=float, default=0.01, help="Generation temperature")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,help='tp')
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Inference batch size (adjust based on GPU memory)")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max new tokens to generate")
    parser.add_argument("--max_context_token_num", type=int, default=4096, help="Max context token number")
    parser.add_argument("--max_session", type=int, default=8192, help="Max Session length")
    return parser.parse_args()

def batch_process(pipe, dataset, args,gen_config):
    
    prompts = []
    valid_queries = []
    
    for query in tqdm(dataset, desc="Preparing batches"):
        try:
            img_paths = query["images"]
            image = [load_image(img_path) for img_path in img_paths]
            question = query["messages"][0]["content"].replace("<image>", "{IMAGE_TOKEN}").strip()
            
            prompts.append((question, image))
            valid_queries.append(query)
        except Exception as e:
            print(f"Skipping {query} due to error: {str(e)}")

    results = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i+args.batch_size]
        responses = pipe(batch_prompts, gen_config=gen_config)
        
        for j, response in enumerate(responses):
            result = {
                "images": valid_queries[i+j]["images"],
                "predict": response.text.strip(),
                'generate_token_len':response.generate_token_len,
                'input_token_len':response.input_token_len,
                'finish_reason':response.finish_reason
            }
            results.append(result)
    
    return results

def main():
    args = parse_args()
    

    engine_config = TurbomindEngineConfig(
        tp=args.tensor_parallel_size,

        max_context_token_num=args.max_context_token_num,
        session_len=args.max_session,
    )
    
    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_p=0.8,
        top_k=40,
        do_sample=True,
        max_new_tokens=args.max_new_tokens,
    )


    pipe = pipeline(
        model_path=args.model_name_or_path,
        backend_config=engine_config
    )


    with open(args.input_path, "r") as f:
        dataset = json.load(f)


    results = batch_process(pipe, dataset, args, gen_config)

    save_dir = os.path.dirname(args.save_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(args.save_name, "w") as f:
        for result in results:
            json_line = json.dumps(result, ensure_ascii=False) 
            f.write(json_line + "\n") 

    pipe.close()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
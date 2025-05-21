import os
import json
import time
from io import BytesIO
os.environ["DISABLE_VERSION_CHECK"] = "1"
import fire
import asyncio
import base64
import concurrent
from tqdm import tqdm
import sglang as sgl
from sglang.test.test_utils import is_in_ci
from llamafactory.hparams import get_infer_args
from transformers import Seq2SeqTrainingArguments
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.model import load_tokenizer
from llamafactory.data.parser import get_dataset_list
from sglang.srt.managers.io_struct import GenerateReqInput
from PIL import Image
from sglang.srt.openai_api.protocol import ChatCompletionRequest
from sglang.srt.server_args import ServerArgs
from sglang.srt.openai_api.adapter import v1_chat_generate_request


if is_in_ci():
    import patch


def main(
    model_name_or_path: str,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 4096,
    max_samples: int = None,
    preprocessing_num_workers: int = 128,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.01,
    top_p: float = 0.001,
    top_k: int = 1,
    max_new_tokens: int = 3000,
    repetition_penalty: float = 1.0,
    parallel_sample_num: int = 1,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    image_resolution: int = 512 * 512,
    gpu_memory_utilization: float = 0.9,
    disable_cuda_graph: bool = False,
):
    time1 = time.time()
    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=preprocessing_num_workers,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            trust_remote_code=True
        )
    )
    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    
    # import IPython; IPython.embed(); 

    def preprocess_sample(sample, tokenizer, template_obj, image_resolution):
        if sample["images"]:
            multi_modal_data = template_obj.mm_plugin._regularize_images(sample["images"], image_resolution=image_resolution)
            
            new_multi_modal_data = []
            # convert to bytes
            for mm_data in multi_modal_data:
                mm_data : Image.Image
                
                # Convert image to bytes
                buffered = BytesIO()
                mm_data.save(buffered, format="png",)
                img_bytes = buffered.getvalue()
                
                # Encode bytes to base64 string
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                img_base64 = f"data:image/png;base64,{img_base64}"
                
                new_multi_modal_data.append(img_base64)
            
        else:
            new_multi_modal_data = None

        input_data = {"prompt_token_ids": sample["input_ids"], "multi_modal_data": new_multi_modal_data}

        return input_data
    
    total_dataset_len = len(dataset_module["train_dataset"])
    # total_dataset_len = 100
    inputs = [None] * total_dataset_len
    with concurrent.futures.ThreadPoolExecutor(max_workers=preprocessing_num_workers) as executor:
        futures = {
            executor.submit(preprocess_sample, dataset_module["train_dataset"][idx], tokenizer, template_obj, image_resolution): idx
            for idx in range(total_dataset_len)
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Preprocessing"):
            idx = futures[future]  # Retrieve the original index
            input_data = future.result()
            inputs[idx] = input_data

    final_input_ids = [x["prompt_token_ids"] for x in inputs]
    final_image_data = [x["multi_modal_data"] for x in inputs]

    # import IPython; IPython.embed();

    engine_kwargs = {
        "model_path": model_args.model_name_or_path,
        "context_length": int(1.5 * cutoff_len),
        "tp_size": tensor_parallel_size,
        "dp_size": data_parallel_size,
        "mem_fraction_static": gpu_memory_utilization,
        "disable_cuda_graph": disable_cuda_graph,
        "chunked_prefill_size": -1,
        "log_level": "INFO",
        "disable_radix_cache": True
    }

    llm = sgl.Engine(**engine_kwargs)
    sampling_params = {
        "n": parallel_sample_num,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty
    }

    time2 = time.time()
    print("Initialization takes", time2-time1, "seconds")

    obj = GenerateReqInput(
        input_ids=final_input_ids,
        image_data=final_image_data,
        sampling_params=sampling_params,
    )

    loop = asyncio.get_event_loop()
    generator = llm.tokenizer_manager.generate_request(obj, None)
    ret = loop.run_until_complete(generator.__anext__())
    
    time3 = time.time()
    print("Generation takes", time3-time2, "seconds")

    # mkdir if not exists
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    with open(save_name, "w", encoding="utf-8") as f:
        for ret_obj in ret:
            f.write(json.dumps({"predict": ret_obj["text"], "meta_info": ret_obj["meta_info"]}, ensure_ascii=False) + "\n")

    print("*" * 70)
    print(f"{len(ret)} generated results have been saved at {save_name}.")
    print("*" * 70)


if __name__ == '__main__':
    fire.Fire(main)

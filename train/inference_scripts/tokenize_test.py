# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import concurrent
os.environ["DISABLE_VERSION_CHECK"] = "1"
import json
from tqdm import tqdm
import fire
from transformers import Seq2SeqTrainingArguments
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import check_version, get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer





if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest


def tokenize_test(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 4096,
    max_samples: int = None,
    preprocessing_num_workers: int = 128,
    vllm_config: str = "{}",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 4096,
    repetition_penalty: float = 1.0,
    pipeline_parallel_size: int = 1,
    image_resolution: int = 512 * 512,
):
    r"""
    Performs batch generation using vLLM engine, which supports tensor parallelism.
    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    check_version("vllm>=0.4.3,<=0.6.5")
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
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
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    model_args.image_resolution = image_resolution
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    data_args.cutoff_len = cutoff_len * 4
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)    
    template_obj.mm_plugin.expand_mm_tokens = True
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "sft", **tokenizer_module)


    
    # # parallel this part of code
    # 6 512x512 images take 2016 tokens
    import IPython; IPython.embed()
    len_of_dataset = len(dataset_module["train_dataset"])
    all_lens = []
    for i in tqdm(range(len_of_dataset), total=len_of_dataset):
        input_ids_sample = dataset_module["train_dataset"][i]['input_ids']
        len_of_input_ids = len(input_ids_sample)
        all_lens.append(len_of_input_ids)
    print(all_lens)
    
    tokenizer.decode(input_ids_sample, skip_special_tokens=False)
    tokenizer.encode("A black car ([3.14, -20.54, 1.81, 4.33, 1.69, -15]) is seen in the front of ego car slightly far from ego car (around 20.78 meters away).\n", add_special_tokens=False)  # 62 tokens...
    
    # analysis all lengths. max, min, mean, std, 25%, 50%, 75%
    import numpy as np
    all_lens = np.array(all_lens)
    print("Max:", all_lens.max())
    print("Min:", all_lens.min())
    print("Mean:", all_lens.mean())
    print("Std:", all_lens.std())
    print("25%:", np.percentile(all_lens, 25))
    print("50%:", np.percentile(all_lens, 50))
    print("75%:", np.percentile(all_lens, 75))
    print("90%:", np.percentile(all_lens, 90))
    print("95%:", np.percentile(all_lens, 95))
    print("99%:", np.percentile(all_lens, 99))
    print("99.9%:", np.percentile(all_lens, 99.9))
    
        # decoded_input_ids = tokenizer.decode(input_ids_sample, skip_special_tokens=False)


if __name__ == "__main__":
    fire.Fire(tokenize_test)
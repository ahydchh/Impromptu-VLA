# https://platform.openai.com/docs/guides/batch
# {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
# {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
# The official documentation states that organizing data into batch format can save half the cost.

import json
from openai import OpenAI
import base64
import os

client = OpenAI(api_key="you need to use your own openai key for evaluation on your local machine")


def generate_batch_json_one(agent, image, prompt, system_prompt=None, custom_id="request-1"):
    if not image.startswith('http'):
        image = imgPath2imgUrl(image)
    request_data = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": agent,
            "messages": [
                {"role": "system", "content": system_prompt if system_prompt else "You are an autonomous driving assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image,
                            }
                        },
                    ],
                }

            ],
            "max_tokens": 3000
        }
    }
    return request_data


def get_image_path(image_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, image_name)
    return image_path


def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def imgPath2imgUrl(image_path):
    """Convert image path to base64 URL"""
    return f"data:image/jpeg;base64,{encode_image(get_image_path(image_path))}"


def generate_multiple_jsons(requests):
    json_list = []
    for i, request in enumerate(requests):
        json_data = generate_batch_json_one(
            agent=request.get("agent"),
            image=request.get("image"),
            prompt=request.get("prompt"),
            system_prompt=request.get("system_prompt"),
            custom_id=f"request-{i + 1}"  # Automatically generate custom_id
        )
        json_list.append(json_data)
    return json_list


def write_to_jsonl(data, filename="batchinput.jsonl"):
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")  # Write each JSON object on a new line


if __name__ == "__main__":
    # Define parameters for multiple requests
    requests = [
        {
            "agent": "gpt-4o-mini",
            "image": "data/img/1.png",
            "prompt": "What's in this image?",
            "system_prompt": "You are a professional assistant. Extract the answer for the user into JSON data."
        },
        {
            "agent": "gpt-4o-mini",
            "image": "data/img/2.png",
            "prompt": "Describe the scene in this image.",
            "system_prompt": "You are a professional assistant. Provide a detailed description of the image."
        },
        {
            "agent": "gpt-4o-mini",
            "image": "data/img/3.jpeg",
            "prompt": "Is there any text in this image? If so, extract it.",
        }
    ]

    # Generate multiple JSON data
    multiple_jsons = generate_multiple_jsons(requests)

    # Write to JSON Lines file
    write_to_jsonl(multiple_jsons, filename="batchinput.jsonl")

    print("JSON Lines file generated: batchinput.jsonl")

    # mind it quit
    quit()

    batch_input_file = client.files.create(
        file=open("batchinput.jsonl", "rb"),
        purpose="batch"
    )

    print(batch_input_file)

    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )

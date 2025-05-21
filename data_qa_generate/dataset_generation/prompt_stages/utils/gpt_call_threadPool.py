# https://platform.openai.com/docs/guides/text-generation
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64

# Initialize OpenAI client
# client = OpenAI(api_key="you need to use your own openai key for evaluation on your local machine")
client = OpenAI(api_key="")


def external_request_one(agent, image, prompt, mode="integrate", system_prompt=None):
    if not image.startswith("http"):
        image = imgPath2imgUrl(image)
    if mode == "analyze_image":
        # Analyze image content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image,
                        },
                    },
                ],
            }
        ]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        completion = client.chat.completions.create(
            model=agent,
            messages=messages,
        )
        usage_info = completion.usage
        print(f"Total tokens used: {usage_info.total_tokens}")
        return completion.choices[0].message.content

    elif mode == "generate_json":
        # Generate JSON data
        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt
                    if system_prompt
                    else "Extract output answer into JSON data."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model=agent,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "email_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "output": {
                                "description": "The answer output",
                                "type": "string",
                            },
                            "additionalProperties": False,
                        },
                    },
                },
            },
        )
        total_tokens = response.usage.total_tokens
        print(f"Total tokens used: {total_tokens}")
        return response.choices[0].message.content

    elif mode == "integrate":
        # Integrate image analysis and JSON generation
        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt
                    if system_prompt
                    else "Extract the answer for the user into JSON data."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image,
                        },
                    },
                ],
            },
        ]

        response = client.chat.completions.create(
            model=agent,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "email_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "output": {
                                "description": "The answer output in JSON format",
                                "type": "string",
                            },
                            "additionalProperties": False,
                        },
                    },
                },
            },
        )
        total_tokens = response.usage.total_tokens
        print(f"Total tokens used: {total_tokens}")
        return response.choices[0].message.content

    else:
        return "Unsupported mode type."


def external_request(requests):
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_request = {}
        for request in requests:
            # Submit the request to the executor
            future = executor.submit(external_request_one, **request)
            future_to_request[future] = request

        for future in as_completed(future_to_request):
            request = future_to_request[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Request {request} generated an exception: {e}")
                results.append(None)
    return results


def get_image_path(image_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, image_name)
    return image_path


def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def imgPath2imgUrl(image_path):
    """Convert image path to base64 URL"""
    return f"data:image/jpeg;base64,{encode_image(get_image_path(image_path))}"


if __name__ == "__main__":
    # Example usage
    requests = [
        {
            "agent": "gpt-4o-mini",
            "image": "data/img/1.png",  # This will be converted to a base64 URL
            "prompt": "What's in this image?",
        },
        {
            "agent": "gpt-4o-mini",
            "image": "data/img/1.png",  # This will be converted to a base64 URL
            "prompt": "What's in this image?",
            "mode": "integrate",
            "system_prompt": "You are a professional assistant. Extract the answer for the user into JSON data.",
        },
        # Add more requests here, mode default "integrate"，
        # and system_prompt can be None
        # {
        #     "agent": "gpt-4o-mini",
        #     "image": "data/img/1.png",  # This will be converted to a base64 URL
        #     "prompt": "What's in this image?",
        # }
    ]

    results = external_request(requests)
    print("all results:")
    print(results)
    for i, result in enumerate(results):
        print(f"Result {i + 1}: {result}")

# output:
# Total tokens used: 1289
# Total tokens used: 1348
# all results:
# ['{"output":"{\\"scene\\":\\"A winding road next to a steep, green hillside.\\",\\"details\\":{\\"road\\":\\"Curved asphalt road\\",\\"vegetation\\":\\"Lush greenery on both sides, including trees and shrubbery\\",\\"rocks\\":\\"Exposed rocky formations along the hill\\",\\"roadside\\":\\"Reflective markers or signs present along the edge of the road\\"}}"}', '{"output":"{\\"description\\":\\"The image shows a scenic view from a car dashboard, depicting a winding road surrounded by lush green trees and foliage. The road curves to the right, and there are guardrails on the right side, indicating a mountainous or hilly area. The atmosphere appears to be bright and sunny, suggesting a pleasant day for driving.\\",\\"elements\\":{\\"foreground\\":\\"Car dashboard with a steering wheel section visible.\\",\\"road\\":\\"Curved roadway extending into the distance.\\",\\"vegetation\\":\\"Thick greenery and trees on either side of the road.\\",\\"other_features\\":\\"No vehicles or pedestrians are visible in the image.\\"}}"}']
# Result 1: {"output":"{\"scene\":\"A winding road next to a steep, green hillside.\",\"details\":{\"road\":\"Curved asphalt road\",\"vegetation\":\"Lush greenery on both sides, including trees and shrubbery\",\"rocks\":\"Exposed rocky formations along the hill\",\"roadside\":\"Reflective markers or signs present along the edge of the road\"}}"}
# Result 2: {"output":"{\"description\":\"The image shows a scenic view from a car dashboard, depicting a winding road surrounded by lush green trees and foliage. The road curves to the right, and there are guardrails on the right side, indicating a mountainous or hilly area. The atmosphere appears to be bright and sunny, suggesting a pleasant day for driving.\",\"elements\":{\"foreground\":\"Car dashboard with a steering wheel section visible.\",\"road\":\"Curved roadway extending into the distance.\",\"vegetation\":\"Thick greenery and trees on either side of the road.\",\"other_features\":\"No vehicles or pedestrians are visible in the image.\"}}"}

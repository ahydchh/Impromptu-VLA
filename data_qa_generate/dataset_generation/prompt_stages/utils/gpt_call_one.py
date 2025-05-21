# https://platform.openai.com/docs/guides/text-generation
import os
from openai import OpenAI
import base64

# Initialize OpenAI client
# client = OpenAI(api_key="you need to use your own openai key for evaluation on your local machine")
client = OpenAI(
    api_key="")


def external_request(agent, image, prompt, mode="integrate", system_prompt=None):
    if not image.startswith('http'):
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
                        }
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
        return completion.choices[0].message.content

    elif mode == "generate_json":
        # Generate JSON data
        messages = [
            {
                "role": "system",
                "content": system_prompt if system_prompt else "Extract output answer into JSON data."
            },
            {
                "role": "user",
                "content": prompt
            }
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
                                "description": "answer output",
                                "type": "string"
                            },
                            "additionalProperties": False
                        }
                    }
                }
            },
        )
        return response.choices[0].message.content

    elif mode == "integrate":
        # Integrate image analysis and JSON generation
        messages = [
            {
                "role": "system",
                "content": system_prompt if system_prompt else "Extract the answer for the user into JSON data."
            },
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
                                "description": "The description of the picture that appears in the output",
                                "type": "string"
                            },
                            "additionalProperties": False
                        }
                    }
                }
            },
        )
        return response.choices[0].message.content

    else:
        return "Unsupported mode type."


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


if __name__ == "__main__":
    # Example usage
    # 1. Analyze image content
    # result_analyze = external_request(
    #     # agent="gpt-4o",
    #     agent="gpt-4o-mini",
    #     image="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
    #     prompt="What's in this image?",
    #     mode="analyze_image",
    #     system_prompt="You are a helpful assistant that describes images in detail."
    # )
    # print("Analyze Image Result:", result_analyze)
    #
    # # 2. Generate JSON data
    # result_json = external_request(
    #     agent="gpt-4o",
    #     agent = "gpt-4o-mini",
    #     image="path/to/image",  # Image parameter is unused in this mode
    #     prompt="Feeling stuck? Send a message to help@mycompany.com.",
    #     mode="generate_json",
    #     system_prompt="You extract email addresses into JSON data."
    # )
    # print("Generate JSON Result:", result_json)

    # 3. Integrated functionality（）
    result_integrate = external_request(
        # agent="gpt-4o",
        agent="gpt-4o-mini",
        image="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        # image="data/img/1.png",
        prompt="What's in this image?",
        mode="integrate",  # default
        system_prompt="You are a professional assistant. Extract the answer for the user into JSON data."
        # system_prompt:defult None (written in advance)
    )

    print("Integrate Result:", result_integrate)

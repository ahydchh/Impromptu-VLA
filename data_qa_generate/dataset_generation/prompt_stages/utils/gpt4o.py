import os
import requests
import base64
import json

# Configuration
API_KEY = os.environ["AZURE_OPENAI_API_KEY"]

def external_request(agent="gpt-4o", image="path/to/image", prompt="your prompt"):
    # 对于Azure Openai:gpt型号是在创建endpoint时就被选择的，所以这里不需要指定
    # 对于Openai:可以代码动态切换
    encoded_image = base64.b64encode(open(image, 'rb').read()).decode('ascii')

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

    # Payload for the request
    payload = {
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
            },
            {
            "type": "text",
            "text": prompt
            },
        ]
        }
    ],
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 2000
    }

    ENDPOINT = "https://zhaohanggpt4v.openai.azure.com/openai/deployments/gpt4o/chat/completions?api-version=2024-02-15-preview" # ENTER THE ENDPOINT HERE

    # Send request
    # retry logic can be added here
    i = 0
    # if the time is too long, retry
    while i<20:
        try:
            response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=30)
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
            break
        except requests.RequestException as e:
            print(f"Failed to make the request. Error: {e}")
            i += 1
            print("Retrying...")
            continue
    
    # Handle the response as needed (e.g., print or process)
    data = response.json()
    print(data["usage"])
    return data['choices'][0]['message']['content']

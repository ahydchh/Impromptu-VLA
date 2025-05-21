import os
import json


dataset_info_path = "dataset_info.json"
new_dataset_info_path = "dataset_info_new.json"
# add your file path
json_dir = ""


if os.path.exists(dataset_info_path):
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
else:
    dataset_info = {}


for filename in os.listdir(json_dir):
    if filename.endswith(".json"):  
        file_path = os.path.join(json_dir, filename)

       
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_content = json.load(f)
                if not json_content or (isinstance(json_content, list) and len(json_content) == 0):
                    print(f"Skipping {filename}: JSON content is empty.")
                    continue 
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
            continue  

        base_name = os.path.splitext(filename)[0]

        dataset_info[base_name] = {
            "file_name": file_path,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }

with open(new_dataset_info_path, "w", encoding="utf-8") as f:
    json.dump(dataset_info, f, indent=4, ensure_ascii=False)

print(f"Updated {new_dataset_info_path} with new JSON files from {json_dir}.")

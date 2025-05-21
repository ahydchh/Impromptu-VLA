
# You need to rewrite the code to regenerate the dataset info file.

import os
import json

# Define the base directory for the dataset files
base_dir = "data/nuscenes"

# Define the default formatting and tags
default_formatting = "sharegpt"
default_columns = {"messages": "messages", "images": "images"}
default_tags = {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
}

# Initialize the dataset info dictionary
dataset_info = {}

# Iterate through the files in the base directory
for file_name in os.listdir(base_dir):
    # Only process JSON files
    if file_name.endswith(".json"):
        # Extract the dataset name (without file extension)
        dataset_name = file_name.replace(".json", "")
        
        # Add an entry to the dataset info dictionary
        dataset_info[dataset_name] = {
            "file_name": f"nuscenes/{file_name}",
            "formatting": default_formatting,
            "columns": default_columns,
            "tags": default_tags
        }

# Output file path
output_file = os.path.join("data/dataset_info.json")

# Write the dataset info dictionary to a JSON file
with open(output_file, "w") as f:
    json.dump(dataset_info, f, indent=4)

print(f"Dataset info file has been generated at: {output_file}")

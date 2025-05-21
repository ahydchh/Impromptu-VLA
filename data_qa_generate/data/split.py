import json
import os
import sys

def split_json_list(file_path, parts=4):

    with open(file_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        
        return

    total = len(data)
    chunk_size = (total + parts - 1) // parts 

    base_name, ext = os.path.splitext(file_path)
    for i in range(parts):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        chunk = data[start:end]
        output_file = f"{base_name}_{i+1}{ext}"
        with open(output_file, 'w') as f_out:
            json.dump(chunk, f_out, indent=2)
        

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("python split_json.py your_file.json")
    else:
        split_json_list(sys.argv[1], parts=4)

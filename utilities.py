import json
import re

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
        

def extract_text_before_number(text):
    # Utilise une expression régulière pour capturer tout ce qui précède le premier chiffre
    # match = re.match(r"([a-zA-Z_]+)", text)
    match = re.match(r"([a-zA-Z]+)", text)
    if match:
        return match.group(1)
    return None
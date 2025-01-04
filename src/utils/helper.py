import yaml
from typing import Dict, Any

def string_to_dict(param_str):
    param_dict = {}
    # Split the string by commas to get key-value pairs
    pairs = param_str.split(',')
    for pair in pairs:
        # Split each pair by '=' to separate key and value
        key, value = pair.split('=')
        # Remove any leading/trailing spaces and convert the value to the right type
        key = key.strip()
        value = value.strip()
        # Check if the value is an integer or float, otherwise keep it as a string
        if value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # keep the value as string if it cannot be converted to int/float
        param_dict[key] = value
    return param_dict

def load_model_search_space(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['parameters']['model_search_space']
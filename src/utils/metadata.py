import json
import os 

def save_metadata(metadata: dict, filename: str):
    """Save metadata to a JSON file

    Args:
        metadata (dict): metadata as a dictionary
        filename (str): filename to be saved
    """
    filepath = os.path.join('model_artifact', filename)
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=4)
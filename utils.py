import yaml
import os
import shutil

def load_yaml(file_path: str):
    with open(file_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            return None

def load_configs(filename: str):
    configs = load_yaml(filename)
    return configs

def create_or_overwrite_directory(path):
    # Check if the directory already exists
    if os.path.exists(path):
        # Remove the existing directory and all its contents
        shutil.rmtree(path)
    
    # Create the new directory
    os.makedirs(path)
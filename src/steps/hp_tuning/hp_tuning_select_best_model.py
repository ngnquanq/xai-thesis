import importlib
import yaml
import json
from utils.helper import string_to_dict
from sklearn.base import ClassifierMixin
import pickle


def hp_tuning_select_best_model(train_config_path) -> ClassifierMixin:
    # Load the train configuration from YAML
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)

    # Extract the best model string
    
    # Load the hp_tuning_result.json file
    with open('model_artifact/training/hp_tuning_result.json', 'r') as f:
        hp_tuning_results = json.load(f)
    
    # Find the model with the highest best_score
    best_model_info = max(
        (model_info 
         for result in hp_tuning_results 
         for model_info in result.values()),
        key=lambda x: x['best_score']
    )
    
    # Extract the best model string
    best_model_str = best_model_info['best_model']
    
    
    # Extract the class name (e.g., DecisionTreeClassifier)
    class_name = best_model_str.split('(')[0].strip()  # Get the class name before the '('

    # Find the corresponding model package in the model_search_space
    model_package = None
    for model_key, model_info in train_config['parameters']['model_search_space'].items():
        if model_info['model_class'] == class_name:
            model_package = model_info['model_package']
            break

    if model_package is None:
        raise ValueError(f"Model package for {class_name} not found in the configuration.")

    # Dynamically import the model class
    module = importlib.import_module(model_package)
    model_class = getattr(module, class_name)

    # Extract parameters from the best_model string
    params_str = best_model_str[best_model_str.index('(') + 1:-1]  # Get the parameters inside the parentheses
    #print(params_str)
    params = string_to_dict(param_str=params_str)
    #print(params)

    # Instantiate the model with the parameters
    model_instance = model_class(**params)
    # Load the fitted model using pickle
    if class_name == 'DecisionTreeClassifier':
        with open('model_artifact/training/decision_tree.pkl', 'rb') as pkl_file:
            model_instance = pickle.load(pkl_file)
    elif class_name == 'LogisticRegression':
        with open('model_artifact/training/logistic_regression.pkl', 'rb') as pkl_file:
            model_instance = pickle.load(pkl_file)
    elif class_name == 'SVC':
        with open('model_artifact/training/svm.pkl', 'rb') as pkl_file:
            model_instance = pickle.load(pkl_file)
    elif class_name == 'RandomForestClassifier':
        with open('model_artifact/training/random_forest.pkl', 'rb') as pkl_file:
            model_instance = pickle.load(pkl_file)
    else:
        raise ValueError(f"Model class {class_name} not supported.")

    return model_instance

# Test the hp_tuning_select_best_model function
if __name__ == '__main__':
    try:
        # Specify the path to the train config
        train_config_path = 'configs/train_config.yaml'
        
        # Call the function
        best_model = hp_tuning_select_best_model(train_config_path)
        
        # Print the result
        print("Best model selected:")
        print(f"Model type: {type(best_model).__name__}")
        print(f"Model parameters: {best_model.get_params()}")
        
        print("\nFunction test completed successfully.")
    except Exception as e:
        print(f"An error occurred while testing the function: {str(e)}")



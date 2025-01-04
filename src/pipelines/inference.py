from steps.hp_tuning.hp_tuning_select_best_model import hp_tuning_select_best_model
from steps.etl.data_loader import *
from steps.etl.train_data_preprocessor import *
from joblib import load
import numpy as np 

def inference(train_config_path):
    # Load the best model parameters
    best_model = hp_tuning_select_best_model(train_config_path)

    # Load the fitted model from the PKL file
    class_name = best_model.__class__.__name__  # Assuming the model name can be derived from the class name
    if class_name == 'DecisionTreeClassifier':
        with open('model_artifact/training/decision_tree.pkl', 'rb') as pkl_file:
            model_instance = load(pkl_file)
    elif class_name == 'LogisticRegression':
        with open('model_artifact/training/logistic_regression.pkl', 'rb') as pkl_file:
            model_instance = load(pkl_file)
    elif class_name == 'SVC':
        with open('model_artifact/training/svm.pkl', 'rb') as pkl_file:
            model_instance = load(pkl_file)
    elif class_name == 'RandomForestClassifier':
        with open('model_artifact/training/random_forest.pkl', 'rb') as pkl_file:
            model_instance = load(pkl_file)
    else:
        raise ValueError(f"Model class {class_name} not supported.")
    # Load the data
    dataset, target, random_state = data_loader(random_state=42, is_inference=True, inference_data="data/synthetic_data.csv")
    dataset_trn, dataset_tst, pipeline = train_data_preprocessor(dataset_tst=dataset)

    print(dataset)
    print(target)
    # Perform inference (predict) instead of fitting
    prediction = model_instance.predict(dataset_tst)  # Use predict instead of fit
    
    # Evaluate the model
    evaluating = model_instance.score(dataset_tst, target)
    
    return evaluating

if __name__ == '__main__':
    try:
        # Specify the path to the train config
        train_config_path = 'configs/train_config.yaml'
        
        # Call the function
        evaluating = inference(train_config_path)
        
        # Print the result
        print("Model evaluation score:")
        print(evaluating)
        
        print("\nFunction test completed successfully.")
    except Exception as e:
        print(f"An error occurred while testing the function: {str(e)}")


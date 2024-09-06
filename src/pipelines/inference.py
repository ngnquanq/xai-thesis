from steps.hp_tuning.hp_tuning_select_best_model import hp_tuning_select_best_model
from steps.etl.data_loader import data_loader

def inference(train_config_path):
    # Load the best model
    best_model = hp_tuning_select_best_model(train_config_path)

    # Load the data
    dataset, target, random_state = data_loader(random_state=42, is_inference=True, inference_data="data/synthetic_data.csv")
    
    # Fit the model
    prediction = best_model.fit(dataset,target)
    
    # Evaluate the model
    evaluating = best_model.score(dataset, target)
    
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


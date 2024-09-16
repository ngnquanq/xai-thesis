from steps.hp_tuning.hp_tuning_select_best_model import hp_tuning_select_best_model
from steps.etl.data_loader import *
from steps.etl.inference_data_preprocessor import *
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

def inference(train_config_path):
    # Load the best model
    best_model = hp_tuning_select_best_model(train_config_path)

    # Load the data
    dataset, target, random_state = data_loader(random_state=42, is_inference=True, inference_data="data/synthetic_data.csv")
    print(dataset.columns)
    inference_dataframe = inference_data_preprocessor(data=dataset)
    X_test = inference_dataframe.drop('Churn', axis=1)
    y_test = inference_dataframe['Churn']
    
    # Fit the model
    #prediction = best_model.fit(dataset,target)
    
    # Evaluate the model
    evaluating = best_model.score(X_test, y_test)
    # Calculate additional evaluation metrics    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate scores
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create a dictionary of scores
    scores = {
        'accuracy': accuracy,
        'f1_score': f1,
        'recall': recall,
        'precision': precision
    }
    
    # Print the scores
    print("\nAdditional evaluation metrics:")
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
    
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


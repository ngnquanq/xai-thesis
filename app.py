import gradio as gr
import pandas as pd
import shap
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset and model
data = pd.read_csv(r".\notebooks\telco\telecom_data_preprocessed.csv")
data = data.drop(columns="Unnamed: 0")
X = data.drop(columns="Churn")
y = data["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_upsampled, y_train_upsampled = smote.fit_resample(X_train, y_train)

# Reset the indices of X_test and y_test to ensure alignment
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Load the YAML file
with open('model_artifact/training/telco/oversampling.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Get the list of models
models = config["models"]

# Extract hyperparameters for Random Forest model
os_random_forest_params = next(model['hyperparameters'] for model in models if model['name'] == 'Random Forest')
os_random_forest_threshold = 0.4449

# Print the hyperparameters
print("Random Forest Parameters:", os_random_forest_params)
print(f"Random Forest Threshold: {os_random_forest_threshold} ")

# Build model and evaluate
xai_model = RandomForestClassifier(**os_random_forest_params)
trained_xai_model = xai_model.fit(X_train_upsampled, y_train_upsampled)

# Load SHAP explainer
with open(r'.\model_artifact\training\telco\telco_explainer.pkl', 'rb') as model_file:
    explainer_object = pickle.load(model_file)

# Function to create a SHAP force plot and display the instance info, true label, and predicted probabilities for both classes
def generate_shap_plot(selected_class):
    # Filter the dataset based on the user-specified class
    if selected_class == "Churn":
        class_indices = y_test[y_test == 1].index  # Indices for churn class
    else:
        class_indices = y_test[y_test == 0].index  # Indices for non-churn class
    
    # Take a random sample from the filtered dataset
    i = random.choice(list(class_indices))  # Use list to ensure correct random choice
    instance = X_test.loc[[i]]  # Use loc to ensure correct selection based on index
    
    # Get the true label from y_test
    true_label = y_test.loc[i]
    
    # Get the predicted probabilities for both classes
    predicted_proba = trained_xai_model.predict_proba(instance)[0]  # Array with probabilities for both classes
    
    # Generate the SHAP force plot for the selected instance
    #shap_values = explainer_object.shap_values(instance)  # Get SHAP values for the specific instance
    shap.initjs()
    plt.figure()
    shap_plot = shap.force_plot(explainer_object.base_values[0][0], explainer_object.values[i,:,0], instance, matplotlib=True, text_rotation=15, figsize=(15,3))  # Adjust to use expected value and shap_values for class 1 (Churn)
    
    # Save the plot to a file and return the file path for Gradio
    plot_path = "shap_force_plot.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    # Create a DataFrame with instance information, true label, and predicted probabilities for both classes
    instance_df = pd.DataFrame(instance)
    instance_df['True Label'] = true_label
    instance_df['Predicted Probability (Non-Churn)'] = predicted_proba[0]  # Probability for class 0
    instance_df['Predicted Probability (Churn)'] = predicted_proba[1]      # Probability for class 1
    
    return plot_path, instance_df

# Create Gradio interface with class selection
interface = gr.Interface(
    fn=generate_shap_plot,
    inputs=gr.Radio(choices=["Churn", "Non-Churn"], label="Select Class"),  # User selects class (Churn or Non-Churn)
    outputs=["image", "dataframe"],  # Outputs SHAP plot and DataFrame with instance info
    title="Random SHAP Force Plot with Instance Info and Prediction",
    description="Select Churn or Non-Churn to generate a SHAP force plot for a random instance, including the true label and predicted probabilities.",
    live=False
)

# Launch the Gradio app
interface.launch(share=True)

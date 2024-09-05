from sdv.datasets.local import load_csvs
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer
import pandas as pd
import os
import yaml  # Add this import at the top

class Maker:
    def __init__(self, synthesizer_class):
        self.data_path = 'data/telecom_churn.csv'
        self.output_path = 'data/synthetic_data.csv'
        self.metadata = SingleTableMetadata()
        self.df = pd.read_csv(self.data_path)
        self.metadata.detect_from_dataframe(self.df)
        self.synthesizer_class = synthesizer_class  # Store the synthesizer class
        self.metadata.update_columns_metadata(
            column_metadata={
                    "Voice mail plan": {"sdtype": "categorical"}
            }
        )

        # Load parameters from the YAML config
        with open('configs/synthesize_config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def save_synthesizer(self, filename):
        """Save the synthesizer to a specified file path."""
        self.synthesizer_class.save(f"model_artifact/synthesizer/{filename}")

    def load_synthesizer(self, filename):
        """Load a synthesizer from a specified file path."""
        filepath = os.path.join('model_artifact', 'synthesizer', filename)
        return self.synthesizer_class.load(filepath)  # Load using the stored synthesizer class

    def validate_metadata(self):
        return self.metadata.validate_data(data=self.df)

    def create_data(self, num_rows):
        if self.synthesizer_class not in [GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer]:
            raise ValueError("Invalid synthesizer class")

        # Select parameters based on synthesizer type
        if self.synthesizer_class == GaussianCopulaSynthesizer:
            parameters = self.config['gaussian_parameters']
        else:  # For CTGAN and CopulaGAN
            parameters = self.config['gan_parameters']

        self.synthesizer_class = self.synthesizer_class(self.metadata, **parameters)  # Pass parameters to the synthesizer
        self.synthesizer_class.fit(self.df)

        synthetic_data = self.synthesizer_class.sample(num_rows=num_rows)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        synthetic_data.to_csv(self.output_path, index=False)
        return synthetic_data

# Example usage:
if __name__ == "__main__":
    maker = Maker(GaussianCopulaSynthesizer)  # Initialize the Maker class with the synthesizer class
    synthetic_data = maker.create_data(num_rows=10)  # Generate synthetic data

    # Save the synthesizer
    maker.save_synthesizer(filename='GaussianCopular_synthesizer.pkl')

    # Load the synthesizer dynamically
    loaded_synthesizer = maker.load_synthesizer('GaussianCopular_synthesizer.pkl')
    print("Synthesizer loaded successfully.")

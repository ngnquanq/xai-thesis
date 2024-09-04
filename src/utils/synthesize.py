from sdv.datasets.local import load_csvs
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer
import pandas as pd

class Maker:
    def __init__(self):
        self.data_path = 'data/telecom_churn.csv'
        self.output_path = 'data/synthetic_data.csv'
        self.metadata = SingleTableMetadata()
        self.df = pd.read_csv(self.data_path)
        self.metadata.detect_from_dataframe(self.df)

    def create_data(self, synthesizer_class, num_rows):
        if synthesizer_class not in [GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer]:
            raise ValueError("Invalid synthesizer class")

        synthesizer = synthesizer_class(self.metadata)
        synthesizer.fit(self.df)

        synthetic_data = synthesizer.sample(num_rows=num_rows)
        
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        synthetic_data.to_csv(self.output_path, index=False)
        return synthetic_data

# Example usage:
# maker = Maker('data/telecom_churn.csv', 'data/synthetic_data.csv')
# synthetic_data = maker.create_data(GaussianCopulaSynthesizer)
# print(synthetic_data)
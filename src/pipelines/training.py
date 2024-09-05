import yaml
from typing import List, Optional, Any, Dict
import random

from steps import (
    data_loader,
    model_evaluator,
    model_trainer,
    train_data_preprocessor,
    train_data_splitter,
    hp_tuning_select_best_model,
    hp_tuning_single_search,
)
from logging import getLogger

logger = getLogger(__name__)

def load_model_search_space(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['parameters']['model_search_space']

def training(
    model_search_space: Dict[str, Any],
    test_size: float = 0.2,
    min_train_accuracy: float = 0.8,
    min_test_accuracy: float = 0.8,
    preprocess: Optional[bool] = True,
    random_state: int = 42
):
    ################ ETL Stage #################
    raw_data, target, _ = data_loader(random_state=random_state)
    dataset_trn, dataset_tst = train_data_splitter(dataset=raw_data, test_size=test_size)
    dataset_trn, dataset_tst, preprocess_pipeline = train_data_preprocessor(dataset_trn=dataset_trn,
                                                                           dataset_tst=dataset_tst)
    
    ################ Hyper parameter tuning ###############
    hp_tuning_final = []
    search_steps_prefix = "hp_tuning_search_"
    for config_name, model_search_configuration in model_search_space.items():
        step_name = f"{search_steps_prefix}{config_name}"
        model_name = f"{config_name}"
        model_best_params = hp_tuning_single_search(
            model_package=model_search_configuration["model_package"],
            model_class=model_search_configuration["model_class"],
            search_grid=model_search_configuration["search_grid"],
            dataset_trn=dataset_trn,
            dataset_tst=dataset_tst,
            target=target
        )
        hp_tuning_final.append({model_name: model_best_params})
        
        # Save model best parameters to JSON file
        import json
        with open(f'model_artifact/training/{model_name}.json', 'w') as json_file:
            json.dump(model_best_params, json_file)

if __name__ == '__main__':
    # Load the model search space from the YAML file
    model_search_space = load_model_search_space('configs/train_config.yaml')

    # Call the training function with the loaded model search space
    training(model_search_space)
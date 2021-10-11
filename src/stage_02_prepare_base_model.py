import argparse
import os
from src.utils.common import read_yaml_file, create_directories
from src.utils.model import get_VGG16_model, prepare_full_model
from tqdm import tqdm
import logging


logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def prepare_base_model(config_path: str, params_path: str) -> None:
    config = read_yaml_file(config_path)
    params = read_yaml_file(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = config["ARTIFACTS_DIR"]

    base_model_dir = artifacts["BASE_MODEL_DIR"]
    base_model_name = artifacts["BASE_MODEL_NAME"]

    base_model_dir_path = os.path.join(artifacts_dir, base_model_dir)
    create_directories([base_model_dir_path])

    base_model_path = os.path.join(base_model_dir_path, base_model_name)

    base_model = get_VGG16_model()

    full_model = prepare_full_model(base_model)

    updated_base_model_path = os.path.join(base_model_dir_path, artifacts["UPDATED_BASE_MODEL_NAME"])

    full_model.save(updated_base_model_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(">>>>> stage two started <<<<<")
        prepare_base_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(">>>>> stage two completed! base model is created and saved successfully <<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
import argparse
import os
from tqdm import tqdm
import logging
from src.utils.common import read_yaml_file, create_directories
from src.utils.model import load_full_model

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def train_model(config_path: str, params_path: str) -> None:
    config = read_yaml_file(config_path)
    params = read_yaml_file(params_path)

    artifacts = config["artifacts"]
    ### get the untrained full model
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    train_model_dir_path = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    create_directories([train_model_dir_path])

    untrained_full_model_path = os.path.join(artifacts_dir, 
    artifacts["BASE_MODEL_DIR"], artifacts["UPDATED_BASE_MODEL_NAME"])

    model = load_full_model(untrained_full_model_path)


    ###  get the callbacks

    ### get the data to create data generator


    ### train the model



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(">>>>> stage four started <<<<<")
        train_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(">>>>> stage four completed! training completed <<<<<n")
    except Exception as e:
        logging.exception(e)
        raise e
import argparse
import os
import shutil
from types import prepare_class
from tqdm import tqdm
import logging


logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def prepare_base_model(config_path):
    pass

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(">>>>> stage two started <<<<<")
        prepare_base_model(config_path=parsed_args.config)
        logging.info(">>>>> stage two completed! base model is created and saved successfully <<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
import argparse
import os
from tqdm import tqdm
import logging
from src.utils.common import read_yaml_file, create_directories, copy_files

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def get_data(config_path):
    config = read_yaml_file(config_path)

    source_data_dirs = config["source_data_dirs"]
    local_data_dirs = config["local_data_dirs"]

    N = len(source_data_dirs)
    for source_data_dir, local_data_dir in tqdm(zip(source_data_dirs, local_data_dirs), total=N, desc="copying directory:"):
        create_directories([local_data_dir])
        copy_files(source_data_dir, local_data_dir)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(">>>>> stage one started <<<<<")
        get_data(config_path=parsed_args.config)
        logging.info(">>>>> stage one completed! all the data are saved in local <<<<<n")
    except Exception as e:
        logging.exception(e)
        raise e
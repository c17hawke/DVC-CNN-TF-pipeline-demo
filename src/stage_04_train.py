import argparse
import os
from tqdm import tqdm
import logging
from src.utils.common import read_yaml_file, create_directories
from src.utils.model import load_full_model, get_unqique_path_to_save_model
from src.utils.callbacks import get_callbacks
from src.utils.data_management import train_valid_generator

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def train_model(config_path: str, params_path: str) -> None:
    """function to train the model and save it into

    Args:
        config_path (str): path to config file
        params_path (str): path to params file
    """
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
    callback_dir_path = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])
    callbacks = get_callbacks(callback_dir_path)

    ### get the data to create data generator
    train_generator, valid_generator = train_valid_generator(
        data_dir=artifacts["DATA_DIR"],
        IMAGE_SIZE=tuple(params["IMAGE_SIZE"][:-1]),
        BATCH_SIZE=params["BATCH_SIZE"],
        do_data_augmention=params["AUGMENTATION"]
    )

    ### train the model
    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size

    model.fit(
        train_generator, 
        validation_data=valid_generator, 
        epochs=params["EPOCHS"], 
        steps_per_epoch=steps_per_epoch, 
        validation_steps=validation_steps, 
        callbacks=callbacks
        )

    ### save the trained model
    trained_model_dir = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    create_directories([trained_model_dir])

    model_file_path = get_unqique_path_to_save_model(trained_model_dir)
    model.save(model_file_path)

    logging.info(f"trained models is saved at: \n{model_file_path}")



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
import tensorflow as tf
import os
import joblib
import logging
from src.utils.common import get_timestamp



def create_and_save_tensorboard_callback(callbacks_dir: str, tensorboard_log_dir: str) -> None:
    """create and save tensorboard_callbacks as binary for later use

    Args:
        callbacks_dir (str): path to callback_dir
        tensorboard_log_dir (str): path to tensorboard_log_dir
    """
    unique_name = get_timestamp("tb_logs")
    tb_running_log_dir = os.path.join(tensorboard_log_dir, unique_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    tb_callback_filepath = os.path.join(callbacks_dir, "tensorboard_cb.cb")
    joblib.dump(tensorboard_callback, tb_callback_filepath)

    logging.info(f"tensorboard callback is saved at {tb_callback_filepath} as binary file")

def create_and_save_checkpointing_callback(callbacks_dir: str, checkpoint_dir: str) -> None:
    """create and save checkpointing callback

    Args:
        callbacks_dir (str): path to callback_dir
        checkpoint_dir (str): path to checkpoint_dir
    """
    checkpoint_file = os.path.join(checkpoint_dir, "ckpt_model.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file, save_best_only=True
    )

    ckpt_callback_filepath = os.path.join(callbacks_dir, "checkpoint_cb.cb")
    joblib.dump(checkpoint_callback, ckpt_callback_filepath)

    logging.info(f"checkpoint callback is saved at {ckpt_callback_filepath} as binary file")


def get_callbacks(callback_dir_path: str)-> list:
    """load saved callbacks from the callbacks directory

    Args:
        callback_dir_path (str): path to callback_dir

    Returns:
        list: list of callbacks for training
    """
    
    callback_paths = [
        os.path.join(callback_dir_path, pickle_file) for pickle_file in os.listdir(callback_dir_path) if pickle_file.endswith(".cb")
    ]
    callbacks = [
        joblib.load(path) for path in callback_paths
    ]

    logging.info(f"saved callbacks are loaded and now ready to be used")
    
    return callbacks
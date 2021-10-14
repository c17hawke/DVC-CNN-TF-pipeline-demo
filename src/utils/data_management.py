import tensorflow as tf
import logging


def train_valid_generator(
    data_dir: str='data',
    IMAGE_SIZE: tuple=(224,224),
    BATCH_SIZE: int=32,
    do_data_augmention: bool=True) -> tuple:
    """create training validation data generator for training

    Args:
        data_dir (str, optional): path to data directory. Defaults to 'data'.
        IMAGE_SIZE (tuple, optional): image size i.e. height and width in pixels. DefDefaults to (224,224).
        BATCH_SIZE (int, optional): Number of samples per gradient update. Defaults to 32.
        do_data_augmention (bool, optional): specify True false to do data augmentation. Defaults to True.

    Returns:
        tuple: a tuples of train and valid generator
    """
    
    datagenerator_kawargs = dict(
        rescale = 1./255, 
        validation_split=0.20
    )

    dataflow_kawargs = dict(
        target_size = IMAGE_SIZE,
        batch_size = BATCH_SIZE,
        interpolation = "bilinear"
    )

    valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kawargs)

    valid_generator = valid_datagenerator.flow_from_directory(
        directory=data_dir,
        subset="validation",
        shuffle=False,
        **dataflow_kawargs
    )

    if do_data_augmention:
        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
            zoom_range=0.2,
            **datagenerator_kawargs
        )
        logging.info("data augmentation is used for training")
    else:
        train_datagenerator = valid_datagenerator
        logging.info("data augmentation is not used for training")

    train_generator = train_datagenerator.flow_from_directory(
        directory=data_dir, 
        subset="training", 
        shuffle=True, 
        **dataflow_kawargs)

    logging.info("train and valid genrator is created.")
    return train_generator, valid_generator
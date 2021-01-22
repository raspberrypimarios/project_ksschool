"""A simple main file to showcase the template."""
import argparse
import logging.config

from tensorflow.keras import datasets, models, layers, activations, optimizers, losses, metrics, utils

def _download_data():
    train, test = datasets.mist.load.data()
    x_train, y_train = train
    x_test, y_test = test
    return x_train, y_train, x_test, y_test


def _preprocess_data(x, y):
    x = x / 255.0
    y = utils.to_categorical(y)

def train_and_evaluate(batch_size, epochs, job_dir, output_path):
    #Download the data
    x_train, y_train, x_test, y_test = _download_data()

    #Preprocess the data
    x_train, y_train = _preprocess_data(x_train, y_train)
    x_test, y_test = _preprocess_data(x_test, y_test)

    pass

def main():
    """Entry point for your module."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type= int)
    parser.add_argument('--job-dir', default=None, required= False)
    parser.add_argument('--model-output-path')

    args = parser.parse_args()
    batch_size= args.batch_size
    epochs = args.epochs
    job_dir = args.job_dir
    output_path = args.model_output_path

    train_and_evaluate(batch_size, epochs, job_dir, output_path)


if __name__ == "__main__":
    main()

"""A simple main file to showcase the template."""
import argparse
import logging.config

from tensorflow.keras import datasets, models, layers, activations, optimizers, losses, metrics, utils

LOGGER = logging.getLogger()

def _download_data():
    LOGGER.info("Downloading data...")
    train, test = datasets.mnist.load.data()
    x_train, y_train = train
    x_test, y_test = test
    return x_train, y_train, x_test, y_test


def _preprocess_data(x, y):
    LOGGER.info("Preprocessing data...")
    x = x / 255.0
    y = utils.to_categorical(y)
    return x,y

def _build_model():
    m = model.Sequential()

    m.add(layers.Input((28,28), name='my_input_layer'))
    m.add(layers.Flatten())
    m.add(layers.Dense(128, activation=activations.relu))
    m.add(layers.Dense(64, activation=activations.relu))
    m.add(layers.Dense(32, activation=activations.relu))
    m.add(layers.Dense(10, activation=activations.softmax))
    
    return m



def train_and_evaluate(batch_size, epochs, job_dir, output_path):
    #Download the data
    x_train, y_train, x_test, y_test = _download_data()

    #Preprocess the data
    x_train, y_train = _preprocess_data(x_train, y_train)
    x_test, y_test = _preprocess_data(x_test, y_test)

    #Build the model
    model=_build_model()

    #Compile model
    model.compile(loss=losses.categorical_crossentropy(),
                    optimizer= optimizers.Adam(),
                    metrics=[metrics.categorical_accuracy])

    #Train model
    model.fit(x_train, y_train, epochs= epochs, batch_size= batch_size)

    #Evaluate model
    loss_value, accuracy = model.evaluate(x_test, y_test)
    LOGGER.info("  *** LOSS VALUE: %f       ACCURACY: %.4f". % (loss_value, accuracy))


def main():
    """Entry point for your module."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int) #Batch size for training
    parser.add_argument('--epochs', type= int) #Number of empochs
    parser.add_argument('--job-dir', default=None, required=False) #Required
    parser.add_argument('--model-output-path') # To write the model

    args = parser.parse_args()


    batch_size= args.batch_size
    epochs = args.epochs
    job_dir = args.job_dir
    output_path = args.model_output_path

    train_and_evaluate(batch_size, epochs, job_dir, output_path)


if __name__ == "__main__":
    main()

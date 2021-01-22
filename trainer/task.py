"""A simple main file to showcase the template."""
import argparse
import logging.config

def train_and_evaluate(batch_size, epochs, job_dir, output_path):
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

import os

# import utils.image_processor
from utils.config import Config


def main():
    # Load configuration file
    config = Config("utils/config.json")
    # print(config.data_dir)

    # Define input and output directories
    input_dir = config.data_dir
    output_dir = config.output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define image processing pipeline
    pipeline = [
        image_processor.Rotate(input_dir, output_dir, degrees=30),
        image_processor.Crop(input_dir, output_dir, x=50, y=50, width=200, height=200)
        # image_processor.StyleTransfer(input_dir, output_dir, model_path=config.model_path)
    ]

    # Process each image in the input directory using the pipeline
    for filename in os.listdir(input_dir):
        for processor in pipeline:
            processor.process_image(filename)

if __name__ == '__main__':
    main()

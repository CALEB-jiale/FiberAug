import os
import cv2
from utils import image_processor
from utils import text_processor
from utils.config import Config
from matplotlib import pyplot as plt
from tqdm import tqdm

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.waitforbuttonpress(0)

def process(input_dir, output_dir, pipeline, processor_func):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each input file using the pipeline and processor function
    print("Processing:")
    for filename in tqdm(os.listdir(input_dir)):
        for processor in pipeline:
            processor_func(processor, filename)
    print("Finished.")

def process_image(processor, filename):
    processor.process_image(filename)

def process_text(processor, filename):
    processor.process_text(filename)

def main():
    # Load configuration file
    config = Config("config.json")

    # Define input and output directories
    # image_input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.data_dir, "images")
    # image_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.output_dir, "test1")
    text_input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.data_dir, "annotations")
    text_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.output_dir, "test2")

    # # test the code
    # print(os.path.join(image_input_dir, "000.jpg"))
    # image = cv2.imread(os.path.join(image_input_dir, "000.jpg"))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # visualize(image)

    # Define image and text processing pipelines
    # image_pipeline = [
    #     image_processor.Rotate(image_input_dir, image_output_dir, degrees=30),
    #     image_processor.Crop(image_input_dir, image_output_dir, x=50, y=50, width=200, height=200)
    #     # image_processor.StyleTransfer(image_input_dir, image_output_dir, model_path=config.model_path)
    # ]
    text_pipeline = [
        text_processor.DeleteLastChar(text_input_dir, text_output_dir)
        # text_processor.AddNoise(text_input_dir, text_output_dir, noise_type='random'),
        # text_processor.Replace(text_input_dir, text_output_dir, replace_type='synonyms')
        # text_processor.Translate(text_input_dir, text_output_dir, src_lang='en', tgt_lang='fr')
    ]

    # Process images and text using the shared 'process' function
    # process(image_input_dir, image_output_dir, image_pipeline, process_image)
    process(text_input_dir, text_output_dir, text_pipeline, process_text)


if __name__ == '__main__':
    main()

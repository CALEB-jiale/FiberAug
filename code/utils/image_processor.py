import os
import cv2
import numpy as np


class ImageProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

    def process_image(self, image_path):
        # Read an image with OpenCV and convert it to the RGB colorspace
        image = cv2.imread(os.path.join(self.input_path, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Processing of image
        processed_image = self._process(image)

        # Save image
        output_image_path = os.path.join(self.output_path, image_path)
        cv2.imwrite(output_image_path, processed_image)

    def _process(self, image):
        # Add image processing methods here
        # Return the processed image
        return image


class Rotate(ImageProcessor):
    def __init__(self, input_path, output_path, degrees):
        super().__init__(input_path, os.path.join(output_path, "Rotate"))
        self.degrees = degrees

    def _process(self, image):
        # Rotate clockwise by specified angle
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), self.degrees, 1)
        return cv2.warpAffine(image, M, (cols, rows))


class Crop(ImageProcessor):
    def __init__(self, input_path, output_path, x, y, width, height):
        super().__init__(input_path, os.path.join(output_path, "Crop"))
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def _process(self, image):
        # Crop specified area
        return image[self.y:self.y+self.height, self.x:self.x+self.width]


# class StyleTransfer(ImageProcessor):
#     def __init__(self, input_path, output_path, model_path):
#         super().__init__(input_path, os.path.join(output_path, "StyleTransfer"))
#         self.model_path = model_path

#         # Load model
#         self.model = load_model(model_path)

#     def _process(self, image):
#         # Style transfer
#         stylized_image = self.model(image)
#         return stylized_image.numpy()

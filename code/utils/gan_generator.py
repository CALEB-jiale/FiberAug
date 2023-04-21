import torch
from torchvision.utils import save_image


class GANGenerator:
    def __init__(self, generator, device=None):
        # Init function, receiving generator
        self.generator = generator
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Move the generator to the device
        self.generator.to(self.device)

        # Set the generator to evaluation mode
        self.generator.eval()

    def generate(self, n_images=1, latent_dim=100):
        # Generate function which takes input and returns the generated images

        # Create random noise for input to generator
        z = torch.randn(n_images, latent_dim, 1, 1, device=self.device)

        # Generate images from noise
        with torch.no_grad():
            generated_images = self.generator(z)

        return generated_images


# # Usage example
# # Create model and load pre-trained parameters
# generator = Generator(nz=100, generator_feature_size=64, num_channels=3)
# generator.load_state_dict(torch.load('generator.pth'))

# # Create Generator
# generator_wrapper = GANGenerator(generator)

# # Generate images
# generated_images = generator_wrapper.generate(n_images=10, latent_dim=100)

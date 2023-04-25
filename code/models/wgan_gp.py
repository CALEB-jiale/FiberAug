import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz=1000, generator_feature_size=64, num_channels=3):
        super(Generator, self).__init__()
        self.nz = nz
        self.generator_feature_size = generator_feature_size
        self.num_channels = num_channels

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.nz,
                               self.generator_feature_size * 8,
                               4,       # kernel_size
                               1,       # stride
                               0,       # padding
                               bias=False),
            nn.BatchNorm2d(self.generator_feature_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.generator_feature_size * 8,
                               self.generator_feature_size * 4,
                               4, 2, 1,
                               bias=False),
            nn.BatchNorm2d(self.generator_feature_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.generator_feature_size * 4,
                               self.generator_feature_size * 2,
                               4, 2, 1,
                               bias=False),
            nn.BatchNorm2d(self.generator_feature_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.generator_feature_size * 2,
                               self.num_channels,
                               4, 2, 1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, discriminator_feature_size=64, num_channels=3):
        super(Discriminator, self).__init__()
        self.discriminator_feature_size = discriminator_feature_size
        self.num_channels = num_channels

        self.main = nn.Sequential(
            nn.Conv2d(self.num_channels,
                      self.discriminator_feature_size,
                      4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.discriminator_feature_size,
                      self.discriminator_feature_size * 2,
                      4, 2, 1, bias=False),
            nn.BatchNorm2d(self.discriminator_feature_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.discriminator_feature_size * 2,
                      self.discriminator_feature_size * 4,
                      4, 2, 1, bias=False),
            nn.BatchNorm2d(self.discriminator_feature_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.discriminator_feature_size * 4,
                      self.discriminator_feature_size * 8,
                      4, 2, 1, bias=False),
            nn.BatchNorm2d(self.discriminator_feature_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.discriminator_feature_size * 8,
                      1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


# # Set random seed for reproducibility
# manualSeed = 999
# torch.manual_seed(manualSeed)

# # Set device
# device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# # Parameters for the model
# nz = 100  # Size of z latent vector (i.e. size of generator input)
# generator_feature_size = 64  # Size of feature maps in generator
# discriminator_feature_size = 64  # Size of feature maps in discriminator
# num_channels = 3  # Number of channels in the training images

# # Initialize Generator and Discriminator
# netG = Generator(nz, generator_feature_size, num_channels).to(device)
# netD = Discriminator(discriminator_feature_size, num_channels).to(device)

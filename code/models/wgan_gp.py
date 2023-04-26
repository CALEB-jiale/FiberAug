import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=1000, target_image_size=(3, 3024, 4032)):
        super().__init__()

        self.z_dim = z_dim
        self.target_image_size = target_image_size
        self.view_image_size = target_image_size[0] * \
            target_image_size[1] * target_image_size[2]

        self.generator = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.view_image_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.generator(x)
        out = out.view(x.size(0), *self.target_image_size)
        return out


class Discriminator(nn.Module):
    def __init__(self, image_size=(3, 3024, 4032)):
        super().__init__()

        in_features = image_size[0] * image_size[1] * image_size[2]

        self.discriminator = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=256),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=256, out_features=64),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.discriminator(x)

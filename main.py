from generator import Generator
from discriminator import Discriminator
from torchvision.datasets.folder import ImageFolder
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.utils import data
from torchvision import transforms
from torchvision import utils as vutils
import torch

batchSize = 64
imageSize = 64

transform = transforms.Compose([
    transforms.Resize(imageSize),
    transforms.CenterCrop(imageSize),  # Optional: crop center of the images
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

portraits_dataset = ImageFolder(root='data', transform=transform)
portraits_dataloader = torch.utils.data.DataLoader(portraits_dataset, batch_size=batchSize, shuffle=True, num_workers=0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

generator_neural_net = Generator()
generator_neural_net.apply(weights_init)

discriminator_neural_net = Discriminator()
discriminator_neural_net.apply(weights_init)

criterion = nn.BCELoss()
optimizer_discriminator = optim.Adam(params=discriminator_neural_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_generator = optim.Adam(params=generator_neural_net.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(1, 76):
    for i, data in enumerate(portraits_dataloader, 0):
        discriminator_neural_net.zero_grad()

        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0]))
        output = discriminator_neural_net(input)
        error_discriminator_real = criterion(output, target)

        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        fake = generator_neural_net(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = discriminator_neural_net(fake.detach())
        error_discriminator_fake = criterion(output, target)

        error_discriminator = error_discriminator_real + error_discriminator_fake
        error_discriminator.backward()
        optimizer_discriminator.step()

        generator_neural_net.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = discriminator_neural_net(fake)
        generator_error = criterion(output, target)
        generator_error.backward()
        optimizer_generator.step()

        print(f'[{epoch}/25][{i}/{len(portraits_dataloader)}] Loss_D: {error_discriminator.item()} Loss_G: {generator_error.item()}')
        if i % 100 == 0:
            fake = generator_neural_net(noise)
            vutils.save_image(fake.data, f'output/fake/fake_samples_epoch_{epoch:03d}.png', normalize=True)

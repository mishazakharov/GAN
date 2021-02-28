import torch


# TODO: finish this for text generation task
class Generator(torch.nn.Module):

    """ Class for global generator network from Pix2PixHD paper

    References:
        https://arxiv.org/pdf/1711.11585.pdf (Pix2PixHD paper)
    """

    def __init__(self, input_dim=3, output_dim=3, ch=64, n_downsampling=3, n_blocks=9):
        super().__init__()
        self.main = torch.nn.ModuleList()
        self.main.append(torch.nn.Sequential(*[
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_dim, ch, kernel_size=7, padding=0),
            torch.nn.BatchNorm2d(ch),
            torch.nn.ReLU(inplace=True)]
        ))
        # Downsampling path
        for i in range(n_downsampling):
            multiplier = 2 ** i
            self.main.append(torch.nn.Sequential(*[torch.nn.Conv2d(ch * multiplier, ch * multiplier * 2, kernel_size=3, stride=2, padding=1),
                             torch.nn.BatchNorm2d(ch * multiplier * 2), torch.nn.ReLU(inplace=True)]))

        # ResNet processing
        for i in range(n_blocks):
            self.main.append(ResNetBlock(ch * multiplier * 2))

        for i in range(n_downsampling):
            multiplier = 2 ** (n_downsampling - i)
            self.main.append(
                torch.nn.Sequential(*[torch.nn.ConvTranspose2d(ch * multiplier, ch * multiplier // 2,
                                                               kernel_size=3, stride=2, padding=1, output_padding=1),
                                      torch.nn.BatchNorm2d(ch * multiplier // 2), torch.nn.ReLU(inplace=True)])
            )

        self.main.append(torch.nn.Sequential(*[torch.nn.ReflectionPad2d(3),
                                               torch.nn.Conv2d(ch, output_dim, kernel_size=7, padding=0),
                                               torch.nn.Tanh()])
                         )

    def forward(self, x):
        output = x
        for layer in self.main:
            output = layer(output)

        return output


class ResNetBlock(torch.nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.main = torch.nn.Sequential(
            # First layer
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(ch, ch, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            # Second layer
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(ch, ch, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(ch)
        )

    def forward(self, x):
        output = x + self.main(x)

        return output


class Discriminator(torch.nn.Module):

    def __init__(self, image_dim=6, ch=64):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(image_dim, ch, 4, 2, 1)
        self.conv_2 = torch.nn.Conv2d(ch, ch * 2, 4, 2, 1)
        self.batch_norm_1 = torch.nn.BatchNorm2d(ch * 2)
        self.conv_3 = torch.nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)
        self.batch_norm_2 = torch.nn.BatchNorm2d(ch * 4)
        self.conv_4 = torch.nn.Conv2d(ch * 4, ch * 8, (1, 4), 1, (0, 1))
        self.batch_norm_3 = torch.nn.BatchNorm2d(ch * 8)
        self.conv_5 = torch.nn.Conv2d(ch * 8, 1, (1, 4), 1, (0, 1))

    def forward(self, x, y, return_features=False):
        output_1 = torch.nn.functional.relu(self.conv_1(torch.cat((x, y), dim=1)))
        output_2 = torch.nn.functional.relu(self.batch_norm_1(self.conv_2(output_1)))
        output_3 = torch.nn.functional.relu(self.batch_norm_2(self.conv_3(output_2)))
        output_4 = torch.nn.functional.relu(self.batch_norm_3(self.conv_4(output_3)))
        output_5 = self.conv_5(output_4)

        if not return_features:
            return torch.nn.functional.sigmoid(output_5)
        else:
            return torch.nn.functional.sigmoid(output_5), [output_1, output_2, output_3, output_4, output_5]


if __name__ == "__main__":
    dummy_input = torch.rand(1, 3, 64, 512)
    dummy_label = torch.rand(1, 3, 64, 512)
    model = Generator()
    disc = Discriminator()
    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
    output = model(dummy_input)
    m = disc(output, dummy_input)

    a = 1


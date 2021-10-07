import torch


class Generator(torch.nn.Module):

    """ U-Net like Generator
    """

    def __init__(self, ch=64, image_dim=3, interpolation: bool = True) -> None:
        """
        """
        super().__init__()
        # Base attributes
        self.interpolation = interpolation

        # Entry
        self.downsample_1 = torch.nn.Conv2d(image_dim, ch, kernel_size=4, stride=2, padding=1)
        # Main encoder
        self.downsample_2 = DownSample(ch, ch * 2)
        self.downsample_3 = DownSample(ch * 2, ch * 4)
        self.downsample_4 = DownSample(ch * 4, ch * 8)
        self.downsample_5 = DownSample(ch * 8, ch * 8)
        self.downsample_6 = DownSample(ch * 8, ch * 8)
        self.downsample_7 = DownSample(ch * 8, ch * 8)
        self.downsample_8 = DownSample(ch * 8, ch * 8, use_bn=False)

        # Decoding path
        self.upsample_1 = UpSample(ch * 8, ch * 8)
        self.upsample_2 = UpSample(ch * 8 * 2, ch * 8)
        self.upsample_3 = UpSample(ch * 8 * 2, ch * 8)
        self.upsample_4 = UpSample(ch * 8 * 2, ch * 8)
        self.upsample_5 = UpSample(ch * 8 * 2, ch * 4)
        self.upsample_6 = UpSample(ch * 4 * 2, ch * 2)
        self.upsample_7 = UpSample(ch * 2 * 2, ch)
        self.upsample_8 = UpSample(ch * 2, image_dim, use_bn=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the whole neural network
        Args:
            x: four-dimensional tensor representing an inpute image/s
        Returns:
            output: three-dimensional tensor with predictions
        """
        # Encoder
        x1 = self.downsample_1(x)
        x2 = self.downsample_2(x1)
        x3 = self.downsample_3(x2)
        x4 = self.downsample_4(x3)
        x5 = self.downsample_5(x4)
        x6 = self.downsample_6(x5)
        x7 = self.downsample_7(x6)
        x8 = self.downsample_8(x7)

        # Decoder
        output = torch.nn.functional.dropout(self.upsample_1(x8), 0.5, training=True)
        output = torch.cat((output, x7), dim=1)
        output = torch.nn.functional.dropout(self.upsample_2(output), 0.5, training=True)
        output = torch.cat((output, x6), dim=1)
        output = torch.nn.functional.dropout(self.upsample_3(output), 0.5, training=True)
        # No more dropouts!
        output = torch.cat((output, x5), dim=1)
        output = self.upsample_4(output)
        output = torch.cat((output, x4), dim=1)
        output = self.upsample_5(output)
        output = torch.cat((output, x3), dim=1)
        output = self.upsample_6(output)
        output = torch.cat((output, x2), dim=1)
        output = self.upsample_7(output)
        output = torch.cat((output, x1), dim=1)
        output = self.upsample_8(output)

        return torch.nn.functional.tanh(output)


class DownSample(torch.nn.Module):

    """ Downsample unit which is just conv + relu + batch norm
    """

    def __init__(self, in_ch: int, out_ch: int, inplace=False, use_bn=True) -> None:
        super().__init__()
        self.downsample = torch.nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.relu = torch.nn.LeakyReLU(0.2, inplace=inplace)
        if use_bn:
            self.batch_norm = torch.nn.BatchNorm2d(out_ch)
        else:
            self.batch_norm = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the downsample unit

        Args:
            x: feature map
        Returns:
            output: downsampled feature map
        """
        output = self.relu(x)
        output = self.downsample(output)
        output = self.batch_norm(output)

        return output


class UpSample(torch.nn.Module):

    """ Upsample unit which is just interpolation/transposed convolution + lrelu + batch norm
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 interpolation: bool = False,
                 use_bn=True) -> None:
        super().__init__()
        if interpolation:
            self.up_sample = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.up_sample = torch.nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1)

        if use_bn:
            self.batch_norm = torch.nn.BatchNorm2d(out_ch)
        else:
            self.batch_norm = torch.nn.Identity()

        self.leaky_relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the upsample unit

        Args:
            x: feature map for upsampling

        Returns:
            output: upsampled feature map
        """
        output = self.leaky_relu(x)
        output = self.up_sample(output)
        output = self.batch_norm(output)

        return output


class GeneratorHD(torch.nn.Module):

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
        self.conv_4 = torch.nn.Conv2d(ch * 4, ch * 8, 4, 1, 1)
        self.batch_norm_3 = torch.nn.BatchNorm2d(ch * 8)
        self.conv_5 = torch.nn.Conv2d(ch * 8, 1, 4, 1, 1)

    def forward(self, x, y, return_features=False):
        output_1 = torch.nn.functional.relu(self.conv_1(torch.cat((x, y), dim=1)))
        output_2 = torch.nn.functional.relu(self.batch_norm_1(self.conv_2(output_1)))
        output_3 = torch.nn.functional.relu(self.batch_norm_2(self.conv_3(output_2)))
        output_4 = torch.nn.functional.relu(self.batch_norm_3(self.conv_4(output_3)))
        output_5 = self.conv_5(output_4)

        features = None if not return_features else [output_1, output_2, output_3, output_4, output_5]

        return torch.nn.functional.sigmoid(output_5), features


# class Discriminator(torch.nn.Module):
#
#     def __init__(self, image_dim=6, ch=64):
#         super().__init__()
#         self.conv_1 = torch.nn.Conv2d(image_dim, ch, 4, 2, 1)
#         self.conv_2 = torch.nn.Conv2d(ch, ch * 2, 4, 2, 1)
#         self.batch_norm_1 = torch.nn.BatchNorm2d(ch * 2)
#         self.conv_3 = torch.nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)
#         self.batch_norm_2 = torch.nn.BatchNorm2d(ch * 4)
#         self.conv_4 = torch.nn.Conv2d(ch * 4, ch * 8, 4, 1, 1)
#         self.batch_norm_3 = torch.nn.BatchNorm2d(ch * 8)
#         self.conv_5 = torch.nn.Conv2d(ch * 8, 1, 4, 1, 1)
#
#     def forward(self, x, y, return_features=False):
#         output = torch.nn.functional.relu(self.conv_1(torch.cat((x, y), dim=1)))
#         output = torch.nn.functional.relu(self.batch_norm_1(self.conv_2(output)))
#         output = torch.nn.functional.relu(self.batch_norm_2(self.conv_3(output)))
#         output = torch.nn.functional.relu(self.batch_norm_3(self.conv_4(output)))
#         output = torch.nn.functional.sigmoid(self.conv_5(output))
#
#         return output


if __name__ == "__main__":
    dummy_input = torch.rand(1, 3, 256, 256)
    dummy_label = torch.rand(1, 3, 256, 256)
    model = Discriminator()
    model_1 = Generator()

    print(sum([p.numel() for p in model_1.parameters() if p.requires_grad]))
    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
    output = model(dummy_input, dummy_label)
    print(output.shape)

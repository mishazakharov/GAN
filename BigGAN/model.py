import torch


class SpectralNorm(object):

    """ Class for performing spectral normalization on layer's weights

    References:
        https://arxiv.org/pdf/1802.05957.pdf (SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS)
        https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
        (I used it for getting some intuition about the concept)
    """

    def __init__(self, n_outputs, n_iterations=1, norm=2):
        self.n_iterations = n_iterations
        self.norm = norm

        self.register_buffer("U_vector", torch.rand(1, n_outputs))

    def compute_weight(self):
        W_matrix = self.weight.view(self.weight.shape[0], -1)
        for _ in range(self.n_iterations):
            spectral_norm, u_vector = self._power_iterations(W_matrix)

        return self.weight / spectral_norm

    def _power_iterations(self, W_matrix):
        with torch.no_grad():
            V_vector = torch.nn.functional.normalize(torch.matmul(self.U_vector, W_matrix), p=self.norm)
            U_vector = torch.nn.functional.normalize(torch.matmul(V_vector, W_matrix.T), p=self.norm)
            spectral_norm = torch.matmul(torch.matmul(U_vector, W_matrix), V_vector.T).squeeze()
            # So that U_vector does not point at any operation in backward graph
            self.U_vector.data = U_vector

            return spectral_norm, U_vector


class Linear(SpectralNorm, torch.nn.Linear):

    def __init__(self, in_features, out_features, bias=True, n_iterations=1, norm=2):
        torch.nn.Linear.__init__(self, in_features, out_features, bias=bias)
        SpectralNorm.__init__(self, out_features, n_iterations=n_iterations, norm=norm)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.compute_weight(), bias=self.bias)


class Conv2d(SpectralNorm, torch.nn.Conv2d):

    def __init__(self, input_channels, output_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, n_iterations=1, norm=2):
        torch.nn.Conv2d.__init__(self, input_channels, output_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, bias=bias)
        SpectralNorm.__init__(self, output_channels, n_iterations=n_iterations, norm=norm)

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.compute_weight(), bias=self.bias, stride=self.stride,
                                          padding=self.padding, dilation=self.dilation, groups=self.groups)


class BatchNorm(torch.nn.Module):

    def __init__(self, channels, y_dim=148):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(channels, affine=False)
        self.linear = Linear(y_dim, channels * 2)

    def forward(self, x, y):
        output = self.batch_norm(x)

        y = self.linear(y)
        gamma, beta = y.chunk(2, dim=1)
        gamma, beta = gamma.unsqueeze(2).unsqueeze(2), beta.unsqueeze(2).unsqueeze(2)

        output = gamma * output + beta

        return output


class SelfAttentionBlock(torch.nn.Module):

    """
    References:
        https://arxiv.org/pdf/1805.08318.pdf (Self-Attention Generative Adversarial Networks)
        https://arxiv.org/pdf/1711.07971.pdf (Non-local Neural Networks)
    """

    def __init__(self, input_channels, k=8):
        super().__init__()
        self.f_layer = Conv2d(input_channels, input_channels // k, kernel_size=1, stride=1)
        self.g_layer = Conv2d(input_channels, input_channels // k, kernel_size=1, stride=1)
        self.h_layer = Conv2d(input_channels, input_channels, kernel_size=1, stride=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1, 1))

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        key = self.f_layer(x).view(b, -1, h * w).permute(0, 2, 1)
        query = self.g_layer(x).view(b, -1, h * w)
        attention_weights = self.softmax(torch.bmm(key, query))
        value = self.h_layer(x).view(b, -1, h * w)
        output = torch.bmm(attention_weights, value.permute(0, 2, 1)).view(b, -1, h, w)

        output = self.gamma * output + x

        return output


class GeneratorBlock(torch.nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, y_dim=148):
        super().__init__()
        self.conv_1 = Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu_1 = torch.nn.ReLU()
        self.relu_2 = torch.nn.ReLU()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.batch_norm_1 = BatchNorm(input_channels, y_dim=y_dim)
        self.batch_norm_2 = BatchNorm(output_channels, y_dim=y_dim)
        # self.residual_conv = torch.nn.Identity()
        # if input_channels != output_channels:
        self.residual_conv = Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):
        output = self.batch_norm_1(x, y)
        output = self.relu_1(output)
        output = self.upsample(output)
        output = self.conv_1(output)

        output = self.batch_norm_2(output, y)
        output = self.relu_2(output)
        output = self.conv_2(output)

        x = self.upsample(x)
        x = self.residual_conv(x)

        return output + x


class Generator(torch.nn.Module):

    def __init__(self, n_classes=1000, ch=96):
        super().__init__()
        self.linear_y = torch.nn.Embedding(n_classes, 128)
        self.linear_z = Linear(20, 4 * 4 * ch * 16)
        self.main_sequence = torch.nn.Sequential(
            GeneratorBlock(16 * ch, 16 * ch),
            GeneratorBlock(16 * ch, 8 * ch),
            GeneratorBlock(8 * ch, 4 * ch),
            GeneratorBlock(4 * ch, 2 * ch),
            SelfAttentionBlock(2 * ch),
            GeneratorBlock(2 * ch, 1 * ch)
        )
        self.bn = torch.nn.BatchNorm2d(1 * ch)
        self.to_RGB_layer = Conv2d(1 * ch, 3, kernel_size=1, stride=1)
        self.tanh = torch.nn.Tanh()

    def forward(self, latent_vector, y):
        latent_vectors = torch.split(latent_vector, 20, dim=1)
        y_embedding = self.linear_y(y)
        output = self.linear_z(latent_vectors[0]).view(latent_vector.shape[0], -1, 4, 4)
        index = 1
        for layer in self.main_sequence:
            if isinstance(layer, GeneratorBlock):
                conditional_vector = torch.cat((latent_vectors[index], y_embedding), dim=1)
                index += 1
                output = layer(output, conditional_vector)
            else:
                output = layer(output)

        output = self.bn(output)
        output = torch.nn.functional.relu(output)
        output = self.to_RGB_layer(output)

        output = self.tanh(output)

        return output


class DiscriminatorBlock(torch.nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, downsample=True):
        super().__init__()
        self.conv_1 = Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu_1 = torch.nn.ReLU()
        self.relu_2 = torch.nn.ReLU()
        self.downsample = torch.nn.Identity()
        if downsample:
            self.downsample = torch.nn.AvgPool2d(2)

        self.residual_conv = Conv2d(input_channels, output_channels, kernel_size=1, stride=1)

    def forward(self, x):
        output = self.relu_1(x)
        output = self.conv_1(output)
        output = self.relu_2(output)
        output = self.conv_2(output)
        output = self.downsample(output)

        x = self.residual_conv(x)
        x = self.downsample(x)

        return output + x


class Discriminator(torch.nn.Module):

    def __init__(self, n_classes=1000, ch=64, image_dim=3):
        super().__init__()
        self.main_sequence = torch.nn.Sequential(
            DiscriminatorBlock(image_dim, 1 * ch),
            SelfAttentionBlock(1 * ch),
            DiscriminatorBlock(1 * ch, 2 * ch),
            DiscriminatorBlock(2 * ch, 4 * ch),
            DiscriminatorBlock(4 * ch, 8 * ch),
            DiscriminatorBlock(8 * ch, 16 * ch),
            DiscriminatorBlock(16 * ch, 16 * ch, downsample=False)
        )
        self.relu = torch.nn.ReLU()

        # Do not spectral normalize this layer for more stable training!
        self.embedding_layer = torch.nn.Embedding(n_classes, 16 * ch)
        self.projection_layer = Linear(16 * ch, 1)

    def forward(self, x, y):
        # They call it 'h' in the paper, so...
        h = self.main_sequence(x)
        h = torch.sum(self.relu(h), dim=[2, 3])
        output = self.projection_layer(h)
        y_embedding = self.embedding_layer(y)
        output = output + torch.sum(y_embedding * output, dim=1).unsqueeze(1)

        return output


if __name__ == "__main__":
    device = "cuda:0"
    with torch.no_grad():
        dummy_input = torch.randn(32, 120).to(device)
        dummy_class_id = torch.zeros(32).long().to(device)
        generator = Generator(n_classes=1000, ch=64).to(device)
        disc = Discriminator(n_classes=1000, ch=64).to(device)
        print(sum([p.numel() for p in disc.parameters() if p.requires_grad]))
        print(sum([p.numel() for p in generator.parameters() if p.requires_grad]))

        fake_batch = generator(dummy_input, dummy_class_id)
        fake_predict = disc(fake_batch, dummy_class_id)
        print(fake_predict.mean())
    # print(generator.main_sequence[3])
    # print(sum([p.numel() for p in generator.parameters() if p.requires_grad]))
    # print(sum([p.numel() for p in generator.main_sequence[3].parameters() if p.requires_grad]))
    #
    # print(sum([p.numel() for p in generator.main_sequence[0].conv_1.parameters() if p.requires_grad]))
    # print(sum([p.numel() for p in generator.main_sequence[0].conv_2.parameters() if p.requires_grad]))
    # print(sum([p.numel() for p in generator.main_sequence[0].residual_conv.parameters() if p.requires_grad]))
    #
    # print(sum([p.numel() for p in generator.main_sequence[0].batch_norm_1.parameters() if p.requires_grad]))
    # print(sum([p.numel() for p in generator.main_sequence[0].batch_norm_2.parameters() if p.requires_grad]))

    a = 1

import math
import random

import torch


def equalized_lr(module, name="weight"):
    EqualizedLR.apply(module, name)

    return module


class EqualizedLR(object):

    """
    From the paper -- Progressive Growing Of GAN's For Improved Quality, Stability, And Variation

    Right after you execute apply method on some torch.nn.Module it will delete .weight from parameters and replace it
    with the same tensor but under .weight_orig variable
    When you execute forward on equalized torch.nn.Module it will take .weight_orig multiply it with
    math.sqrt(2 / fan_in) and assign it to a .weight variable and use it for forward
    Optimizer will use this "normalized" output to calculate gradients and update .weight_orig so in next forward usage
    .weight will be scaled version of new updated .weight_orig. That's how it works!

    References:
        https://github.com/rosinality/style-based-gan-pytorch
        https://arxiv.org/pdf/1710.10196.pdf
    """

    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + "_orig")
        fan_in = weight.data.shape[1] * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualizedLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + "_orig", torch.nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


class EqualizedLinearLayer(torch.nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        linear_layer = torch.nn.Linear(channels_in, channels_out)
        linear_layer.weight.data.normal_(mean=0, std=1)
        linear_layer.bias.data.zero_()
        self.linear_layer = equalized_lr(linear_layer)

    def forward(self, x):
        return self.linear_layer(x)


class EqualizedConv2d(torch.nn.Module):

    """ Equalized Convolution Layer
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        conv2d = torch.nn.Conv2d(*args, **kwargs)
        conv2d.weight.data.normal_(mean=0, std=1)
        conv2d.bias.data.zero_()

        self.conv2d = equalized_lr(conv2d)

    def forward(self, x):
        return self.conv2d(x)


class BlurFunctionBackward(torch.autograd.Function):

    """ Separate function for performing backward propagation of gaussian blur operation on activations for faster
    execution on train-time because it does not calculate any kind of gradients!

    References:
        https://github.com/rosinality/style-based-gan-pytorch/blob/b01ffcdcbca6d8bcbc5eb402c5d8180f4921aae4/model.py#L122
    """

    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = torch.nn.functional.conv2d(grad_output, kernel_flip, padding=1, groups=grad_output.shape[1])

        return grad_input

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors
        grad_input = torch.nn.functional.conv2d(grad_output, kernel, padding=1, groups=grad_output.shape[1])

        return grad_input, None, None


class BlurFunction(torch.autograd.Function):

    """ Separate function for performing gaussian blur operation on activations for faster execution on train-time
    because it does not calculate any kind of gradients at backward propagation!

    References:
        https://github.com/rosinality/style-based-gan-pytorch/blob/b01ffcdcbca6d8bcbc5eb402c5d8180f4921aae4/model.py#L144
    """

    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = torch.nn.functional.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors
        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


class BlurLayer(torch.nn.Module):

    """ Basic Gaussian blur layer that is used for low-pass filtering on activations
    """

    def __init__(self, channel):
        super().__init__()
        # Basic 3x3 kernel for performing Gaussian blur operation!
        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.unsqueeze(0).unsqueeze(0)
        weight /= weight.sum()
        self.b = weight.repeat(channel, 1, 1, 1)
        weight_flipped = torch.flip(weight, [2, 3])

        self.register_buffer("weight", weight.repeat(channel, 1, 1, 1))
        self.register_buffer("weight_flip", weight_flipped.repeat(channel, 1, 1, 1))

    def forward(self, x):
        return BlurFunction.apply(x, self.weight, self.weight_flip)


class PixelWiseNormalization(torch.nn.Module):

    """ Normalization Layer for latent code

    References:
         https://arxiv.org/pdf/1710.10196.pdf (4.2)
    """
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ConstantInput(torch.nn.Module):
    def __init__(self, n_channels, size=4):
        super().__init__()
        self.constant_input = torch.nn.Parameter(torch.randn(1, n_channels, size, size))

    def forward(self, x):
        batch_size = x.shape[0]

        return self.constant_input.repeat(batch_size, 1, 1, 1)


class NoiseInput(torch.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.scaling_factor = torch.nn.Parameter(torch.zeros(1, n_channels, 1, 1))

    def forward(self, x, noise):
        return x + self.scaling_factor * noise


class AdaptiveInstanceNormalization(torch.nn.Module):

    """ AdaIN layer
    """

    def __init__(self, input_ch, w_dim=512):
        super().__init__()
        self.instance_normalization = torch.nn.InstanceNorm2d(input_ch)
        # Layer that learns affine transforms for styling on an input
        self.affine_transformation_layer = EqualizedLinearLayer(w_dim, input_ch * 2)
        # Since bias is initialized with zeros we got to adjust it manually
        self.affine_transformation_layer.linear_layer.bias.data[:input_ch] = 1
        self.affine_transformation_layer.linear_layer.bias.data[input_ch:] = 0

    def forward(self, x, style):
        styles = self.affine_transformation_layer(style)
        y_s, y_b = torch.chunk(styles, 2, dim=1)
        output = self.instance_normalization(x)
        output = y_s.unsqueeze(2).unsqueeze(3) * output + y_b.unsqueeze(2).unsqueeze(3)

        return output


class SynthesisNetworkBlock(torch.nn.Module):

    """ Main building block for synthesis network
    """

    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, w_dim=512, first=False, fused=False):
        super().__init__()
        # Noise inputs for stochastic variations
        self.noise_injection_1 = equalized_lr(NoiseInput(output_channels), name="scaling_factor")
        self.noise_injection_2 = equalized_lr(NoiseInput(output_channels), name="scaling_factor")
        # Two convolutions
        if first:
            # In first block you always plug in constant input
            self.convolution_1 = ConstantInput(input_channels, size=4)
            self.convolution_2 = EqualizedConv2d(
                input_channels, output_channels, kernel_size=kernel_size, padding=padding)
        else:
            if fused:
                # Fused is just a transposed convolution instead of interpolation
                self.convolution_1 = torch.nn.Sequential(
                    FusedUpsample(input_channels, output_channels, kernel_size=3, padding=1),
                    BlurLayer(output_channels)
                )
            else:
                self.convolution_1 = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode="nearest"),
                    EqualizedConv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding),
                    BlurLayer(output_channels)
                )
            self.convolution_2 = EqualizedConv2d(
                output_channels, output_channels, kernel_size=kernel_size, padding=padding)
        # AdaIN layers
        self.adaIN_1 = AdaptiveInstanceNormalization(output_channels, w_dim=w_dim)
        self.adaIN_2 = AdaptiveInstanceNormalization(output_channels, w_dim=w_dim)
        # Activations after convolution-noise injection
        self.leaky_relu_1 = torch.nn.LeakyReLU(0.2)
        self.leaky_relu_2 = torch.nn.LeakyReLU(0.2)

    def forward(self, x, style):
        output = self.convolution_1(x)
        gaussian_noise = self.generate_noise_inputs(output)
        output = self.noise_injection_1(output, gaussian_noise)
        output = self.leaky_relu_1(output)
        output = self.adaIN_1(output, style)

        output = self.convolution_2(output)
        output = self.noise_injection_2(output, gaussian_noise)
        output = self.leaky_relu_2(output)
        output = self.adaIN_2(output, style)

        return output

    @staticmethod
    def generate_noise_inputs(sample):
        batch_size, _, w, h = sample.shape
        noise = torch.randn(batch_size, 1, w, h).to(sample.device)

        return noise


class MappingNetwork(torch.nn.Module):

    """ Performs Z-space - W-space mapping
    """

    def __init__(self, z_dim=512, w_dim=512, n_layers=8, normalize=True):
        super().__init__()
        if normalize:
            self.normalization_layer = PixelWiseNormalization()
        else:
            self.normalization_layer = torch.nn.Identity()
        placeholder = [(EqualizedLinearLayer(z_dim, w_dim),
                        torch.nn.LeakyReLU(0.2)) if i == 0 else (
            EqualizedLinearLayer(w_dim, w_dim), torch.nn.LeakyReLU(0.2)) for i in range(n_layers)]
        mapping_layers = list()
        list(map(lambda x: mapping_layers.extend(x), placeholder))
        self.mapping_network_layers = torch.nn.Sequential(*mapping_layers)

    def forward(self, latent_code):
        output = self.normalization_layer(latent_code)
        output = self.mapping_network_layers(output)

        return output


class SynthesisNetwork(torch.nn.Module):

    """ Synthesis network class
    """

    def __init__(self, image_dim=3, fused=True):
        super().__init__()
        # Sequence of main synthesis blocks
        self.synthesis_network = torch.nn.Sequential(
            SynthesisNetworkBlock(512, 512, first=True),  # 4x4
            SynthesisNetworkBlock(512, 512),  # 8x8
            SynthesisNetworkBlock(512, 512),  # 16x16
            SynthesisNetworkBlock(512, 512),  # 32x32
            SynthesisNetworkBlock(512, 256),  # 64x64
            SynthesisNetworkBlock(256, 128, fused=fused),  # 128x128
            SynthesisNetworkBlock(128, 64, fused=fused),  # 256x256
            SynthesisNetworkBlock(64, 32, fused=fused),  # 512x512
            SynthesisNetworkBlock(32, 16, fused=fused)  # 1024x1024
        )
        # Output of the last layer is always converted to RGB using separate 1x1 convolution layer
        self.to_RGB_layers = torch.nn.Sequential(
            EqualizedConv2d(512, image_dim, kernel_size=1, stride=1, padding=0),
            EqualizedConv2d(512, image_dim, kernel_size=1, stride=1, padding=0),
            EqualizedConv2d(512, image_dim, kernel_size=1, stride=1, padding=0),
            EqualizedConv2d(512, image_dim, kernel_size=1, stride=1, padding=0),
            EqualizedConv2d(256, image_dim, kernel_size=1, stride=1, padding=0),
            EqualizedConv2d(128, image_dim, kernel_size=1, stride=1, padding=0),
            EqualizedConv2d(64, image_dim, kernel_size=1, stride=1, padding=0),
            EqualizedConv2d(32, image_dim, kernel_size=1, stride=1, padding=0),
            EqualizedConv2d(16, image_dim, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, w_vectors, pg_step=8, alpha=1):
        # Only binary style mixing is implemented!
        assert len(w_vectors) < 3
        output_last = w_vectors[0]
        w_vector = w_vectors[0]
        # Where to switch from w1 generation to w2 generation
        crossover_point = pg_step + 1
        if len(w_vectors) > 1:
            crossover_point = random.randint(1, pg_step)

        for index, block in enumerate(self.synthesis_network):
            # Switching w vectors - styles!
            if index >= crossover_point:
                w_vector = w_vectors[1]
            output_cur = block(output_last, w_vector)
            # Progressive growing logic
            if index == pg_step:
                break
            output_last = output_cur

        output = self.to_RGB_layers[index](output_cur)
        # PG smooth fade in of a new layer
        if 0 <= alpha < 1:
            image_last = self.to_RGB_layers[index-1](output_last)
            image_last = torch.nn.functional.interpolate(image_last, scale_factor=2, mode="nearest")
            output = (1 - alpha) * image_last + alpha * output

        return output


class StyleBasedGenerator(torch.nn.Module):
    def __init__(self, z_dim=512, w_dim=512, n_layers=8, normalize=True, image_dim=3, fused=True):
        super().__init__()
        self.mapping_network = MappingNetwork(z_dim=z_dim, w_dim=w_dim, n_layers=n_layers, normalize=normalize)
        self.synthesis_network = SynthesisNetwork(image_dim=image_dim, fused=fused)

    def forward(self, noise_vectors, progressive_growing_phase=8, alpha=1):
        # Style mixing
        if not (isinstance(noise_vectors, tuple) or isinstance(noise_vectors, list)):
            noise_vectors = (noise_vectors,)

        w_vectors = list()
        for noise_vector in noise_vectors:
            w_vectors.append(self.mapping_network(noise_vector))

        generated_images = self.synthesis_network(w_vectors, progressive_growing_phase, alpha)
        # generated_images = torch.nn.functional.tanh(generated_images)

        return generated_images


# TODO: write some explanation to it!
class FusedUpsample(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = math.sqrt(2 / fan_in)

        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = torch.nn.functional.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = torch.nn.functional.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class DownsampleBlock(torch.nn.Module):

    """ Basic building block for a discriminator in StyleGAN
    """

    def __init__(self, input_channels, output_channels, kernel_size_1, padding_1,
                 kernel_size_2=None, padding_2=None, fused=False, downsample=True):
        super().__init__()
        if kernel_size_2 is None:
            kernel_size_2 = kernel_size_1
        if padding_2 is None:
            padding_2 = padding_1
        self.convolution_1 = torch.nn.Sequential(
            EqualizedConv2d(input_channels, output_channels, kernel_size=kernel_size_1, padding=padding_1),
        )
        self.activation_1 = torch.nn.LeakyReLU(0.2)
        if downsample:
            if fused:
                self.convolution_2 = torch.nn.Sequential(
                    BlurLayer(output_channels),
                    EqualizedConv2d(output_channels, output_channels, kernel_size=4, stride=2, padding=padding_2),
                )
            else:
                self.convolution_2 = torch.nn.Sequential(
                    BlurLayer(output_channels),
                    EqualizedConv2d(output_channels, output_channels, kernel_size=kernel_size_2, padding=padding_2),
                    torch.nn.AvgPool2d(2),
                )
        else:
            self.convolution_2 = EqualizedConv2d(
                output_channels, output_channels, kernel_size=kernel_size_2, padding=padding_2)

        self.activation_2 = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        output = self.convolution_1(x)
        output = self.activation_1(output)

        output = self.convolution_2(output)
        output = self.activation_2(output)

        return output


class Discriminator(torch.nn.Module):
    def __init__(self, fused=True, use_activations=False):
        super().__init__()
        self.downsampling_blocks = torch.nn.Sequential(
            DownsampleBlock(16, 32, 3, 1, fused=fused),  # 512x512
            DownsampleBlock(32, 64, 3, 1, fused=fused),  # 256x256
            DownsampleBlock(64, 128, 3, 1, fused=fused),  # 128x128
            DownsampleBlock(128, 256, 3, 1, fused=fused),  # 64x64
            DownsampleBlock(256, 512, 3, 1),  # 32x32
            DownsampleBlock(512, 512, 3, 1),  # 16x16
            DownsampleBlock(512, 512, 3, 1),  # 8x8
            DownsampleBlock(512, 512, 3, 1),  # 4x4
            DownsampleBlock(513, 512, 3, 1, 4, 0, downsample=False) # 1x1
        )

        self.from_RGB_layers = self.build_from_RGB_layers(use_activations)
        self.final_projection = EqualizedLinearLayer(512, 1)

        self.use_activations = use_activations

    def forward(self, x, progressive_growing_phase=8, alpha=1):
        output = x
        for index in range(progressive_growing_phase + 1):
            cur_idx = -progressive_growing_phase - 1 + index
            cur_layer = self.downsampling_blocks[cur_idx]
            if index == 0:
                cur_idx_rgb = cur_idx if not self.use_activations else cur_idx * 2
                output = self.from_RGB_layers[cur_idx_rgb](output)
                if self.use_activations:
                    output = self.from_RGB_layers[cur_idx_rgb+1](output)
            # Increasing variation using mini-batch standard deviation!
            if index == progressive_growing_phase:
                averaged_std = output.std(dim=0, unbiased=False).mean()
                # Create feature map with std
                averaged_std = averaged_std.expand(output.shape[0], 1, output.shape[2], output.shape[3])
                output = torch.cat([output, averaged_std], dim=1)

            output = cur_layer(output)
            # PG smooth fade in
            if 0 <= alpha < 1 and index == 0 and cur_idx != -1:
                x = torch.nn.functional.avg_pool2d(x, 2)
                cur_idx_rgb = cur_idx if not self.use_activations else cur_idx * 2
                x = self.from_RGB_layers[cur_idx_rgb](x)
                if self.use_activations:
                    x = self.from_RGB_layers[cur_idx_rgb+1](output)
                output = (1 - alpha) * x + alpha * output

        output = self.final_projection(output.squeeze(2).squeeze(2))
        # output = torch.nn.functional.sigmoid(output)

        return output

    @staticmethod
    def build_from_RGB_layers(use_activations):
        output_channels = [16, 32, 64, 128, 256, 512, 512, 512, 512]
        placeholder = [((EqualizedConv2d(3, oc, kernel_size=1),
                        torch.nn.LeakyReLU(0.2))) if use_activations else (
            EqualizedConv2d(3, oc, kernel_size=1),) for oc in output_channels]
        mapping_layers = list()
        list(map(lambda x: mapping_layers.extend(x), placeholder))

        return torch.nn.Sequential(*mapping_layers)


if __name__ == "__main__":
    dummy_input = torch.FloatTensor(1, 3, 1024, 1024).to("cuda:0")
    disc = Discriminator(use_activations=True).to("cuda:0")
    output = disc(dummy_input, pg_step=8, alpha=0.3)

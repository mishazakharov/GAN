import torch


def get_wasserstein_loss(net_G, net_D, batch=None, noise_vector=None, c_lambda=10, discriminator=True):
    if discriminator:
        # Discriminator real batch
        D_x = net_D(batch).view(-1)
        D_x = D_x.mean() - 0.001 * (D_x ** 2).mean()
        (-D_x).backward()

        # Discriminator fake batch
        fake_batch = net_G(noise_vector)
        D_G_z1 = net_D(fake_batch.detach()).view(-1)
        D_G_z1 = D_G_z1.mean()
        D_G_z1.backward()

        # Calculate gradients on net_D with respect to mixed images
        epsilon = torch.rand(fake_batch.shape[0], 1, 1, 1, device=fake_batch.device, requires_grad=True)
        mixed_images = batch.data * epsilon + fake_batch.data * (1 - epsilon)
        mixed_scores = net_D(mixed_images)
        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_scores.sum(),
            create_graph=True,
        )[0]
        gradient_penalty = ((gradient.view(gradient.shape[0], -1).norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty = c_lambda * gradient_penalty
        gradient_penalty.backward()

        return D_x, D_G_z1, gradient_penalty
    else:
        fake_batch = net_G(noise_vector)
        D_G_z2 = net_D(fake_batch).view(-1)

        G_error = -D_G_z2.mean()
        G_error.backward()

        return G_error, D_G_z2


def get_minimax_loss(net_G, net_D, criterion, label, batch, noise_batch=None, fake_label=None, discriminator=True):
    if discriminator:
        ############################
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        output = net_D(batch).view(-1)
        # Calculate loss on all-real batch
        D_error_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        D_error_real.backward()
        D_x = output.mean().item()

        fake_batch = net_G(noise_batch)
        label.fill_(fake_label)
        output = net_D(fake_batch.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        D_error_fake = criterion(output, label)
        # Calculate the gradients for this batch
        D_error_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        D_error = D_error_real + D_error_fake

        return D_error, D_G_z1, D_x, fake_batch
    else:
        ############################
        # Update G network: maximize log(D(G(z)))
        ###########################
        output = net_D(batch).view(-1)
        # Calculate G's loss based on this output
        G_error = criterion(output, label)
        # Calculate gradients for G
        G_error.backward()
        D_G_z2 = output.mean().item()

        return G_error, D_G_z2


############################ Legacy code
def get_gradient(crit, real, fake, epsilon):
    """
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    """
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    return gradient


def gradient_penalty(gradient):

    gradients_norm = torch.sqrt(torch.sum(gradient ** 2, dim=1) + 1e-12)

    return ((gradients_norm - 1) ** 2).mean()


def get_gen_loss(crit_fake_pred):
    """
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    """
    gen_loss = -crit_fake_pred.mean()

    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    """
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    """
    crit_loss = crit_fake_pred.mean() - crit_real_pred.mean() + c_lambda * gp

    return crit_loss
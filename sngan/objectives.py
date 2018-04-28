from torch import nn
from torch.autograd import Variable
from torch import autograd
import torch.nn.functional as F
import torch


def get_js_gen_loss(gan, discriminator, batch_size, z_size):
    criterion = nn.Softplus().cuda()
    z = Variable(torch.Tensor(batch_size, z_size).normal_(0,1)).cuda()
    prime = gan(z)
    fake_out = discriminator(prime)
    fake_loss = criterion(-fake_out).mean()
    fake_loss.backward()

    return fake_loss


def js_penalty(real, fake, discriminator):
    real = Variable(real.data.cuda(), requires_grad=True)
    fake = Variable(fake.data.cuda(), requires_grad=True)
    real_out = discriminator(real)
    fake_out = discriminator(fake)
    g_r = autograd.grad(outputs=real_out, inputs=real, grad_outputs=torch.ones(real_out.size()).cuda(),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]

    g_f = autograd.grad(outputs=fake_out, inputs=fake, grad_outputs=torch.ones(fake_out.size()).cuda(),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]

    g_r = (g_r.view(g_r.size()[0], -1) ** 2).sum(1).mean()
    g_f = (g_f.view(g_f.size()[0], -1) ** 2).sum(1).mean()

    return 0.5 * (g_r + g_f)


def interpolate_penalty(real, fake, discriminator):
    real = Variable(real.data.cuda(), requires_grad=True)
    fake = Variable(fake.data.cuda(), requires_grad=True)
    alpha = Variable(torch.Tensor(1, 1, 1, 1).uniform_(0, 1)).cuda()
    inter = (1. - alpha) * real + alpha * fake

    inter_out = discriminator(inter)
    g = autograd.grad(outputs=inter_out, inputs=inter, grad_outputs=torch.ones(inter_out.size()).cuda(),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
    return (g.view(g.size()[0], -1) ** 2).sum(1).mean()


def get_js_disc_loss(train_data, gan, discriminator, batch_size, z_size, use_penalty):
    criterion = nn.Softplus().cuda()
    real = Variable(train_data[0].cuda())
    real = 2 * real - 1.
    real_out = discriminator(real)
    real_loss = criterion(-real_out).mean()

    z = Variable(torch.FloatTensor(batch_size, z_size).normal_(0,1).cuda())
    prime = gan(z).detach()
    fake_out = discriminator(prime)
    fake_loss = (criterion(-fake_out) + fake_out).mean()
    loss = real_loss + fake_loss
    if use_penalty:
        loss += js_penalty(real, prime, discriminator)
    loss.backward()
    return real_loss, fake_loss


def get_w_gen_loss(gan, discriminator, batch_size, z_size):
    z = Variable(torch.Tensor(batch_size, z_size).normal_(0,1)).cuda()
    prime = gan(z)
    fake_out = discriminator(prime).mean()
    fake_out.backward()
    return fake_out


def get_w_disc_loss(train_data, gan, discriminator, batch_size, z_size, use_penalty):
    real = Variable(train_data[0].cuda())
    real = 2 * real - 1.
    real_out = discriminator(real).mean()

    z = Variable(torch.FloatTensor(batch_size, z_size).normal_(0,1).cuda())
    prime = gan(z).detach()
    fake_out = discriminator(prime).mean()

    loss = real_out - fake_out
    if use_penalty:
        loss += interpolate_penalty(real, prime, discriminator)
    loss.backward()
    return real_out, fake_out


losses = {
    'jsgan': (get_js_gen_loss, get_js_disc_loss),
    'wgan': (get_w_gen_loss, get_w_disc_loss)
}

import torch
import torch.nn as nn


class RALoss(nn.Module):

    def __init__(self):
        super(RALoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, real_out, fake_out):
        # Label Smoothing
        zeros = torch.rand(real_out.shape).type_as(real_out).uniform_(0, 0.3)
        ones = torch.ones(real_out.shape).type_as(real_out).uniform_(0.7, 1.2)

        a = real_out - torch.mean(fake_out, dim=0)
        b = fake_out - torch.mean(real_out, dim=0)
        loss = (self.loss_fn(a, ones) + self.loss_fn(b, zeros)) * 0.5
        return loss

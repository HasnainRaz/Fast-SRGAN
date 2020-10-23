import torch
import torch.nn as nn


class RALoss(nn.Module):

    def __init__(self):
        super(RALoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, real_out, fake_out):
        ones = torch.ones(real_out.shape).type_as(real_out)
        zeros = torch.zeros(real_out.shape).type_as(real_out)

        a = real_out - torch.mean(fake_out, dim=0)
        b = fake_out - torch.mean(real_out, dim=0)
        loss = (self.loss_fn(a, zeros) + self.loss_fn(b, ones)) * 0.5
        return loss

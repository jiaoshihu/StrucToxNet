
import torch
import torch.nn.functional as F
import torch.nn as nn


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return 1 - F.cosine_similarity(p, z, dim=-1)
    else:
        raise Exception
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, config):
        super(ContrastiveLoss, self).__init__()
        self.margin = config.margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        # euclidean_distance = F.pairwise_distance(output1, output2)
        cos_distance = D(output1, output2)
        # print("ED",euclidean_distance)
        loss_contrastive = torch.mean((1 - label) * torch.pow(cos_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 3))

        return loss_contrastive




def get_loss(logits, labels, criterion):
    labels = labels.long()
    loss = criterion(logits.view(-1, 2), labels.view(-1))
    loss = loss.float().mean()

    # flooding method
    # loss = (loss - 0.2).abs() + 0.2

    # multi-sense loss
    # alpha = -0.1
    # loss_dist = alpha * cal_loss_dist_by_cosine(model)
    # loss += loss_dist

    return loss



class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = nn.functional.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()




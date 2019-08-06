import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class AttrSoftLoss(nn.Module):
    """
    soft margin loss for this
    """
    def __init__(self):
        super(AttrSoftLoss, self).__init__()

    def forward(self, x):
        """
        consider attributes as multiple labels
        carry out multi-label classification loss
        :param x: score and attributes
        :return: loss
        """
        scores, attributes = x
        attr_loss = 0.0

        for i in range(attributes.shape[0]):
            loss_mask = torch.ones(attributes.shape[1]).cuda()
            zeros = (attributes[i, :] == 0).nonzero().cpu().numpy()
            indices = np.random.choice(zeros.squeeze(), int(round(len(zeros) * 0.8)), False)
            loss_mask[indices] = 0

            attr_loss += F.multilabel_margin_loss(attributes[i].unsqueeze(0),
                                                  scores[i].unsqueeze(0),
                                                  weight=loss_mask)
        attr_loss /= attributes.shape[0]
        return attr_loss


class AttrClassifier(nn.Module):
    """
    Linear Classifier
    """
    def __init__(self, args):
        super(AttrClassifier, self).__init__()
        self.in_dim = args.feat_dim * args.crop_height * args.crop_width
        self.num_class = args.supervision['attr']['other']['num_attr']
        self.classifier = nn.Linear(self.in_dim, self.num_class)
        self.sigmoid = nn.Sigmoid()
        self.loss = AttrSoftLoss()

    def forward(self, agg_data):
        """
        forward pipeline, compute loss function
        :param agg_data: refer to ../base_model.py
        :return: loss, acc
        """
        x = agg_data['features']
        attributes = agg_data['attr']
        x = self.classifier(x)
        x = self.sigmoid(x)
        attributes = attributes[:x.shape[0]].long()
        loss = self.loss([x, attributes])
        return loss
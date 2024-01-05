import torch.nn as nn
from KittiAnchors import Anchors


class YODAModel(nn.Module):
    def __init__(self, yoda_classifier, anchors=None):
        super(YODAModel, self).__init__()
        self.yoda_classifier = yoda_classifier
        if anchors is None:
            self.anchors = Anchors()
        else:
            self.anchors = anchors

    def forward(self, x):
        return self.yoda_classifier(x)

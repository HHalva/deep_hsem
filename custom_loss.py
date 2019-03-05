import torch
from torch import nn
import torch.backends.cudnn as cudnn
from poincare_model import PoincareDistance, PoincareDistance2
import pdb
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance

cudnn.fastest = True

pdist = PoincareDistance
edist = PairwiseDistance()

def poincare_dist_to_label_emb(pred_embs, all_embs):
    batch_size = pred_embs.size(0)
    n_emb_dims = pred_embs.size(1)
    n_classes = all_embs.size(0)
    #calculate distance of a predicted embedding to all possible true
    #embedding
    return PoincareDistance.apply(pred_embs.repeat(1,
                                  n_classes).view(-1, n_emb_dims),
                                  all_embs.repeat(batch_size,
                                  1).cuda(non_blocking=True)).view(
                                  batch_size, -1)

def _assert_no_grad(tensor):
    assert not tensor.requires_grad

class PoincareXEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xeloss = nn.CrossEntropyLoss().cuda()

    def forward(self, pred_embs, target_idx, all_embs):
        _assert_no_grad(target_idx)
        _assert_no_grad(all_embs)
        scores = poincare_dist_to_label_emb(pred_embs, all_embs)
        #since smaller distance is good need to inver the exponent in
        #softmax
        neg_scores = -1 * scores
        return self.xeloss(neg_scores, target_idx)

class PoincareSQDistLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_embs, target_embs):
        _assert_no_grad(target_embs)
        return PoincareDistance.apply(pred_embs, target_embs).pow(2).mean()


def euc_dist_to_label_emb(pred_embs, all_embs):
    batch_size = pred_embs.size(0)
    n_emb_dims = pred_embs.size(1)
    n_classes = all_embs.size(0)
    #calculate distance of a predicted embedding to all possible true
    #embedding
    return edist(pred_embs.repeat(1,
                                  n_classes).view(-1, n_emb_dims),
                                  all_embs.repeat(batch_size,
                                  1).cuda(non_blocking=True)).view(
                                  batch_size, -1)


class EmbXEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xeloss = nn.CrossEntropyLoss().cuda()

    def forward(self, pred_embs, target_idx, all_embs):
        _assert_no_grad(target_idx)
        _assert_no_grad(all_embs)
        scores = euc_dist_to_label_emb(pred_embs, all_embs)
        #since smaller distance is good need to inver the exponent in
        #softmax
        neg_scores = -1 * scores
        return self.xeloss(neg_scores, target_idx)



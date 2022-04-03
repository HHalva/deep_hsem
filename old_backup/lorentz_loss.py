import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from poincare_model import EuclideanDistance
import pdb
from torch.autograd import Function
from lorentz import LorentzManifold

cudnn.fastest = True


def _assert_no_grad(tensor):
    assert not tensor.requires_grad

def lorentz_dist_to_label_emb(pred_embs, all_embs):
    batch_size = pred_embs.size(0)
    n_emb_dims = pred_embs.size(1)
    n_classes = all_embs.size(0)
    #calculate distance of a predicted embedding to all possible true
    #embedding
    return LorentzManifold().distance(pred_embs.repeat(1,
                                  n_classes).view(-1, n_emb_dims),
                                  all_embs.repeat(batch_size,
                                  1).cuda(non_blocking=True)).view(
                                  batch_size, -1)

def euc_dist_to_label_emb(pred_embs, all_embs):
    batch_size = pred_embs.size(0)
    n_emb_dims = pred_embs.size(1)
    n_classes = all_embs.size(0)
    #calculate distance of a predicted embedding to all possible true
    #embedding
    return F.pairwise_distance(pred_embs.repeat(1,
                               n_classes).view(-1, n_emb_dims),
                               all_embs.repeat(batch_size,
                               1).cuda(non_blocking=True),
                               keepdim=True).view(
                               batch_size, -1)

class LorentzEmbDist(Function):
    """Custom function which allows to calculate Lorentzian embedding
       distance between predicted and true embedding location.
       WARNING!!!: The backprop here assumes that the second input is 
       the known embedding locations and thus has zero gradients so
       be very careful with how you use this!!!"""
    @staticmethod
    def forward(ctx, pred_embs, all_embs):
        ctx.save_for_backward(pred_embs, all_embs)
        return lorentz_dist_to_label_emb(pred_embs, all_embs)

    def backward(ctx, g):
        pred_embs, all_embs, = ctx.saved_tensors
        with th.enable_grad():
            z = lorentz_dist_to_label_emb(pred_embs, all_embs)
            d_pred = th.autograd.grad(z, pred_embs, g)[0]
            #d_pred.narrow(-1, 0, 1).mul_(-1)
            #d_pred.addcmul_(LorentzManifold().ldot(
            #                               pred_embs, d_pred,
            #                               keepdim=True).expand_as(
            #                               pred_embs), pred_embs)
            return d_pred, th.zeros(1).expand_as(all_embs)

class LorentzXEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xeloss = nn.CrossEntropyLoss().cuda()

    def forward(self, pred_embs, target_idx, all_embs):
        _assert_no_grad(target_idx)
        _assert_no_grad(all_embs)
        #scores = lorentz_dist_to_label_emb(pred_embs, all_embs)
        scores = LorentzEmbDist.apply(pred_embs, all_embs)
        #since smaller distance is good need to invert the exponent in
        #softmax
        neg_scores = -1 * scores
        return self.xeloss(neg_scores, target_idx)

class EuclideanEmbeddingXELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xeloss = nn.CrossEntropyLoss().cuda()

    def forward(self, pred_embs, target_idx, all_embs):
        _assert_no_grad(target_idx)
        _assert_no_grad(all_embs)
        scores = euc_dist_to_label_emb(pred_embs, all_embs)
        #since smaller distance is good need to invert the exponent in
        #softmax
        neg_scores = -1 * scores
        return self.xeloss(neg_scores, target_idx)



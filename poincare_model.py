#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from numpy.random import choice, randint
import torch as th
from torch import nn
from torch.autograd import Function, Variable
from torch.utils.data import Dataset
from collections import defaultdict as ddict
import pdb

eps = 1e-5


class Arcosh(Function):
    def __init__(self, eps=eps):
        super(Arcosh, self).__init__()
        self.eps = eps

    def forward(self, x):
        self.z = th.sqrt(x * x - 1)
        return th.log(x + self.z)

    def backward(self, g):
        z = th.clamp(self.z, min=eps)
        z = g / z
        return z

def grad(x, v, sqnormx, sqnormv, sqdist):
    alpha = (1 - sqnormx)
    beta = (1 - sqnormv)
    z = 1 + 2 * sqdist / (alpha * beta)
    a = ((sqnormv - 2 * th.sum(x * v, dim=-1) + 1) / th.pow(alpha,
      2)).unsqueeze(-1).expand_as(x)
    a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
    z = th.sqrt(th.pow(z, 2) - 1)
    z = th.clamp(z * beta, min=eps).unsqueeze(-1)
    euc_grad = 4 * a / z.expand_as(x)

    return euc_grad

def poincare_grad(p, d_p):
    if d_p.is_sparse:
        p_sqnorm = th.sum(
                p.data[d_p._indices()[0].squeeze()] ** 2, dim=1,
                keepdim=True
                ).expand_as(d_p._values())
        n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
        d_p = spten_t(d_p._indices(), n_vals, d_p.size())
    else:
        p_sqnorm = th.sum(p.data ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)

    return d_p


class PoincareDistance(Function):

    @staticmethod
    def forward(ctx, u, v, boundary=1-eps):
        ctx.save_for_backward(u, v)
        ctx.squnorm = th.clamp(th.sum(u * u, dim=-1), 0, boundary)
        ctx.sqvnorm = th.clamp(th.sum(v * v, dim=-1), 0, boundary)
        ctx.sqdist = th.sum(th.pow(u - v, 2), dim=-1)
        x = ctx.sqdist / ((1-ctx.squnorm) * (1-ctx.sqvnorm))*2 + 1
        # arcosh
        z = th.sqrt(th.pow(x, 2) - 1)
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = grad(u, v, ctx.squnorm, ctx.sqvnorm, ctx.sqdist)
        gv = grad(v, u, ctx.sqvnorm, ctx.squnorm, ctx.sqdist)
        #gu = poincare_grad(u, gu)
        #gv = poincare_grad(v, gv)

        return g.expand_as(gu) * gu, g.expand_as(gv) * gv

class PoincareDistance2(nn.Module):
    def __init__(self, boundary=0.99999):
        super().__init__()
        self.boundary = boundary

    def forward(self, U, V):
        squnorm = th.clamp(th.sum(U.pow(2), dim=1, keepdim=True), 0, self.boundary)
        sqvnorm = th.clamp(th.sum(V.pow(2), dim=1, keepdim=True), 0, self.boundary)
        sqdist = th.sum(U.sub(V).pow(2), dim=1, keepdim=True)
        denominator = th.mul(th.add(th.neg(squnorm), 1),
                                th.add(th.neg(sqvnorm), 1))
        dist = th.add(th.mul(sqdist.div(denominator), 2), 1)
        # arcosh
        z = th.sqrt(dist.pow(2).add(-1))
        return th.log(th.add(dist, z))

class EuclideanDistance(nn.Module):
    def __init__(self, radius=1, dim=None):
        super(EuclideanDistance, self).__init__()

    def forward(self, u, v):
        return th.sum(th.pow(u - v, 2), dim=-1)


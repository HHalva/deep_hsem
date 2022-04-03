import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

import pdb
import random
import shutil
import time
import warnings

#import cv2
#cv2.setNumThreads(0)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from nltk.corpus import wordnet as wn
from poincare import PoincareManifold


#setup arguments
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='21k ImageNet ZSL evaluation')
parser.add_argument('--euclidean', default='False', action='store_true',
                    help='evaluate Euclidean rather than Poincare embedding')
parser.add_argument('-d', '--dim', default=50, type=int,
                    help='dimensionality of embedding')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='size of batches for evaluation')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--emb-dir', default='/home/hermanni/final_embeddings/',
                    type=str, help='location for embedding files')
parser.add_argument('--nn-weights', default='', type=str, metavar='PATH',
                    help='path to traind model weight (default:  none)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg16)')
parser.add_argument('--eval-dir',
                    default='/mnt/fast-data15/datasets/imagenet/fa2011',
                    type=str, help='location for 21k imgnet files')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--cpu', '-c', default=False, action='store_true',
                    help='use CPU rather than GPU')


def main():
    #parse args
    global args
    args = parser.parse_args()

    #GPU setting
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    #import dataset
    if args.euclidean == True:
        emb_name = "euclidean"
    else:
        emb_name = "poincare"
    file_name = str(args.dim)+emb_name[0]+'_nouns.pth'
    embedding = torch.load(args.emb_dir+file_name)
    print("=> loaded label embedding '{}'".format(file_name))

    # check dimension
    if emb_name == "euclidean":
        assert args.dim == embedding['model']['lt.weight'].shape[1]
    elif emb_name == "poincare":
        assert args.dim == embedding['embeddings'].shape[1]

    #change labels from synset names into imagenet format
    synset_list = [wn.synset(i) for i in embedding['objects']]
    offset_list = [wn.ss2of(j) for j in synset_list]
    embedding['objects'] = ['n'+i.split('-')[0] for i in offset_list]

    #load the CNN part of the model 
    print("=> using pre-trained model '{}'".format(args.arch))
    orig_vgg = models.__dict__[args.arch](pretrained=True)

    #change the model to project into desired embedding space
    if emb_name == "euclidean":
        model = EuclidEmbVGG(orig_vgg, args.dim)
    elif emb_name == "poincare":
        model = PoincareEmbVGG(orig_vgg, args.dim)
    model.features = torch.nn.DataParallel(model.features)
    model.to(device, non_blocking=True)

    #load weights from training on 1K classes
    if os.path.isfile(args.nn_weights):
        print("=> loading checkpoint '{}'".format(args.nn_weights))
        checkpoint = torch.load(args.nn_weights)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.nn_weights,
                                                 checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.nn_weights))

    #data loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    eval_dataset = datasets.ImageFolder(args.eval_dir, transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,]))
    eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    #sort embedding to match image labels
    img_labels = eval_dataset.classes
    img2emb_idx = [embedding['objects'].index(i)
                   for i in img_labels]
    if emb_name == "euclidean":
        emb_wgts = embedding['model']['lt.weight'][img2emb_idx]
    elif emb_name == "poincare":
        emb_wgts = embedding['embeddings'][img2emb_idx]
    emb_wgts = emb_wgts.float().cuda(non_blocking=True)
    n_classes = emb_wgts.shape[0]

    #load 21k class distance matrix
    class_distance_mat = torch.load('class_dist_mat.pt').to(device,
            non_blocking=True)
    class_distance_mat = class_distance_mat+torch.t(class_distance_mat)

    #trackers
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':.3f')
    top2 = AverageMeter('Acc@2', ':.3f')
    top5 = AverageMeter('Acc@5', ':.3f')
    top10 = AverageMeter('Acc@10', ':.3f')
    top20 = AverageMeter('Acc@20', ':.3f')

    top1_pos = AverageMeter('Top1+', ':.2f')
    top2_pos = AverageMeter('Top2+', ':.2f')
    top5_pos = AverageMeter('Top5+', ':.2f')
    top10_pos = AverageMeter('Top10+', ':.2f')
    top20_pos = AverageMeter('Top20+', ':.2f')
    top1_neg = AverageMeter('Top1-', ':.2f')
    top2_neg = AverageMeter('Top2-', ':.2f')
    top5_neg = AverageMeter('Top5-', ':.2f')
    top10_neg = AverageMeter('Top10-', ':.2f')
    top20_neg = AverageMeter('Top20-', ':.2f')

    progress = ProgressMeter(
        len(eval_loader),
        [batch_time, top1, top2, top5, top10, top20,
            top1_pos, top2_pos, top5_pos, top10_pos, top20_pos,
            top1_neg, top2_neg, top5_neg, top10_neg, top20_neg],
        prefix='Eval: ')

    #evaluate
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(eval_loader):
            #print(i)
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)

            # compute top-k accuracies
            (prec1, prec2, prec5, prec10, prec20), preds = accuracy(
                    output, emb_wgts, target, (1, 2, 5, 10, 20), emb_name)

            # compute top-k +/- (see paper)
            target_dist_mat = class_distance_mat[target]
            topk_pos, topk_neg = calc_topk_pos_neg(preds, target_dist_mat,
                    (1, 2, 5, 10, 20))
            tkpos_1, tkpos_2, tkpos_5, tkpos_10, tkpos_20 = topk_pos
            tkneg_1, tkneg_2, tkneg_5, tkneg_10, tkneg_20 = topk_neg

            # track evaluation
            top1.update(prec1.item(), images.size(0))
            top2.update(prec2.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))
            top10.update(prec10.item(), images.size(0))
            top20.update(prec20.item(), images.size(0))

            top1_pos.update(tkpos_1.item(), images.size(0))
            top2_pos.update(tkpos_2.item(), images.size(0))
            top5_pos.update(tkpos_5.item(), images.size(0))
            top10_pos.update(tkpos_10.item(), images.size(0))
            top20_pos.update(tkpos_20.item(), images.size(0))

            top1_neg.update(tkneg_1.item(), images.size(0))
            top2_neg.update(tkneg_2.item(), images.size(0))
            top5_neg.update(tkneg_5.item(), images.size(0))
            top10_neg.update(tkneg_10.item(), images.size(0))
            top20_neg.update(tkneg_20.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print evaluation 
            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(i)


#define model classes
class PoincareEmbVGG(nn.Module):
    def __init__(self, vgg_model, n_emb_dims, unfreeze=False):
        super(PoincareEmbVGG, self).__init__()
        self.features = vgg_model.features
        self.fc = nn.Sequential(*list(
                                vgg_model.classifier.children())[:-1])
        self.classifier = nn.Sequential(nn.Linear(4096, n_emb_dims))
        self.dir_func = nn.Linear(n_emb_dims, n_emb_dims, bias=False)
        self.norm_func = nn.Linear(n_emb_dims, 1, bias=False)
        nn.init.uniform_(self.dir_func.weight, -0.001, 0.001)
        nn.init.uniform_(self.norm_func.weight, -0.001, 0.001)

        #default is is to unfreeze classifier i.e. fully connected layers only
        self.unfreeze_features(unfreeze)
        self.unfreeze_fc(True)

    def unfreeze_features(self, unfreeze):
        for p in self.features.parameters():
            p.requires_grad = unfreeze

    def unfreeze_fc(self, unfreeze):
        for p in self.fc.parameters():
            p.requires_grad = unfreeze

    def forward(self, x):
        f = self.features(x)
        if hasattr(self, 'fc'):
            f = f.view(f.size(0), -1)
            f = self.fc(f)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        dir_vec = self.dir_func(y)
        norms_magnitude = self.norm_func(y)
        v = dir_vec.div(torch.norm(dir_vec, dim=1, keepdim=True))
        p = torch.sigmoid(norms_magnitude)
        return p*v


class EuclidEmbVGG(nn.Module):
    def __init__(self, vgg_model, n_emb_dims, unfreeze=False):
        super(EuclidEmbVGG, self).__init__()
        self.features = vgg_model.features
        self.fc = nn.Sequential(*list(
            vgg_model.classifier.children())[:-1])
        self.classifier = nn.Sequential(nn.Linear(4096, n_emb_dims))
        self.dir_func = nn.Linear(n_emb_dims, n_emb_dims, bias=False)
        self.norm_func = nn.Linear(n_emb_dims, 1, bias=False)
        nn.init.uniform_(self.dir_func.weight, -0.001, 0.001)
        nn.init.uniform_(self.norm_func.weight, -0.001, 0.001)

        # default is to unfreeze classifier i.e. fully connected layers
        self.unfreeze_features(unfreeze)
        self.unfreeze_fc(True)

    def unfreeze_features(self, unfreeze):
        for p in self.features.parameters():
            p.requires_grad = unfreeze

    def unfreeze_fc(self, unfreeze):
        for p in self.fc.parameters():
            p.requires_grad = unfreeze

    def forward(self, x):
        f = self.features(x)
        if hasattr(self, 'fc'):
            f = f.view(f.size(0), -1)
            f = self.fc(f)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


def prediction(output, all_embs, k, emb_type):
    with torch.no_grad():
        b_size = output.shape[0]
        embs_d = output.shape[1]
        n_classes = all_embs.shape[0]
        expand_out = output.repeat(1, n_classes).view(-1, embs_d)
        expand_embs = all_embs.repeat(b_size, 1)
        if emb_type == 'poincare':
            dists_to_all = PoincareManifold().distance(expand_out, expand_embs)
        elif emb_type == 'euclidean':
            dists_to_all = F.pairwise_distance(expand_out, expand_embs,
                                               keepdim=True)
        topk_per_batch = torch.topk(dists_to_all.view(b_size, -1),
                                    k=k, dim=1, largest=False)[1]
        if k == 1:
            return topk_per_batch.view(-1)
        return topk_per_batch


def accuracy(output, all_embs, targets, topk=(1,), emb_type='poincare'):
    """Computes the accuracy over the k top predictions for the specified
    values of k"""
    with torch.no_grad():
        maxk = max(topk)
        preds = prediction(output, all_embs, maxk, emb_type)
        batch_size = targets.size(0)
        res = []
        for k in topk:
            i = k
            preds_tmp = preds[:, :i]
            correct_tmp = preds_tmp.eq(
                    targets.view(batch_size, -1).repeat(1, i))
            res.append(torch.sum(correct_tmp).float() / batch_size)
        return res, preds


def calc_topk_pos_neg(preds, target_dist_mat, topk=(1,)):
    """ Evaluation criterion proposed by Brad."""
    with torch.no_grad():
        topk_pos_res = []
        topk_neg_res = []
        max_any_dist_to_target = torch.max(target_dist_mat, 1)[0]
        for k in topk:
            topk_preds = preds[:, :k]
            pred_dist_to_target = target_dist_mat[torch.arange(
                target_dist_mat.shape[0])[:, None], topk_preds]
            min_pred_dist_to_target = torch.min(pred_dist_to_target, 1)[0]
            max_pred_dist_to_target = torch.max(pred_dist_to_target, 1)[0]
            topk_pos_res.append((min_pred_dist_to_target /
                    max_any_dist_to_target).mean())
            topk_neg_res.append((max_pred_dist_to_target /
                    max_any_dist_to_target).mean())
        return topk_pos_res, topk_neg_res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val'+self.fmt+'} ({avg'+self.fmt+'})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits)+ 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def offset_to_label(wnet_offset):
    return wn.of2ss(wnet_offset.split('n')[1]+'-n')


if __name__ == '__main__':
    main()

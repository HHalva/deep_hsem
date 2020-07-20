import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

import pdb
import random
import shutil
import time
import warnings

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
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='size of batches for evaluation')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--emb_dir', default='/home/hermanni/poincare-embeddings/',
                    type=str, help='location for embedding files')
parser.add_argument('--emb_file_name', default=None, type=str, required=True,
                    help='name of the embedding file')
parser.add_argument('--saved_weights', default='', type=str, metavar='PATH',
                    help='path to traind model weight (default:  none)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg16)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

def main():

    #parse args
    global args
    args = parser.parse_args()
    if args.emb_file_name is None:
        raise NameError('args.emb_file_name is not specified')

    #GPU setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    #import dataset
    embedding = torch.load(args.emb_dir+args.emb_file_name)
    print('EMBEDDING TYPE:', embedding['manifold'])
    n_emb_dims = embedding['embeddings'].shape[1]
    args.n_emb_dims = n_emb_dims
    print('NUM OF DIMENSIONS:', n_emb_dims)

    #change labels from synset names into imagenet format
    synset_list = [wn.synset(i) for i in embedding['objects']]
    offset_list = [wn.ss2of(j) for j in synset_list]
    embedding['objects'] = ['n'+i.split('-')[0] for i in offset_list]

    #load the CNN part of the model 
    print("=>using pre-trained model '{}'".format(args.arch))
    orig_vgg = models.__dict__[args.arch](pretrained=True)

    #change the model to project into desired embedding space
    if embedding['manifold'] == 'poincare':
        model = PoincareEmbVGG(orig_vgg, args.n_emb_dims)
    elif embedding['manifold'] == 'euclidean':
        model = EuclidEmbVGG(orig_vgg, args.n_emb_dims)
    model.to(device, non_blocking=True)
    model.features = torch.nn.DataParallel(model.features)

    #load weights from training on 1K classes
    if os.path.isfile(args.saved_weights):
        print("=> loading checkpoint '{}'".format(args.saved_weights))
        checkpoint = torch.load(args.saved_weights)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.saved_weights,
                                                 checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.saved_weights))

    #data loading
    evaldir = '/mnt/fast-data15/datasets/imagenet/mammals'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    eval_dataset = datasets.ImageFolder(evaldir, transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,]))

    eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    #sort embedding to match image labels
    img_labels = eval_dataset.classes
    img2emb_idx = [embedding['objects'].index(i)
                   for i in img_labels]
    emb_wgts = embedding['embeddings'][img2emb_idx]
    emb_wgts = emb_wgts.float().to(device, non_blocking=True)
    n_classes = emb_wgts.shape[0]

    #load 21k class distance matrix
    class_distance_mat = torch.load('class_dist_mat.pt').to(device,
            non_blocking=True)
    class_distance_mat = class_distance_mat+torch.t(class_distance_mat)

    #trackers
    batch_time = AverageMeter('Time', ':6.3f')
    top5_pos_track = AverageMeter('Top5+', ':6.2f')
    top5_neg_track = AverageMeter('Top5-', ':6.2f')
    progress = ProgressMeter(
        len(eval_loader),
        [batch_time, top5_pos_track, top5_neg_track],
        prefix='Eval: ')

    #evaluate
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(eval_loader):
            print(i)
            #if i <= 25329:
            #    continue
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            #compute output
            output = model(images)

            #evaluate
            preds = prediction(output, emb_wgts, 5, embedding['manifold'])
            target_dist_mat = class_distance_mat[target]
            top5_pos, top5_neg = calc_top5_pos_neg(preds, target_dist_mat)

            #track evaluation
            top5_pos_track.update(top5_pos, preds.shape[0])
            top5_neg_track.update(top5_neg, preds.shape[0])

            #measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

        print(
              ' * Top5+ {top5_pos_track.avg: .3f} Top5- {top5_neg_track.avg:.3f}'.format(top5_pos_track=top5_pos_track, top5_neg_track=top5_neg_track))


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


def calc_top5_pos_neg(preds, target_dist_mat):
    with torch.no_grad():
        top5_pos = 0
        top5_neg = 0
        for i in range(preds.shape[0]):
            target_distances = target_dist_mat[i]
            pred_dist_to_target = target_distances[preds[i]]
            min_pred_dist = torch.min(pred_dist_to_target)
            max_pred_dist = torch.max(pred_dist_to_target)
            top5_pos += min_pred_dist / torch.max(target_distances)
            top5_neg += max_pred_dist / torch.max(target_distances)
        top5_pos /= args.batch_size
        top5_neg /= args.batch_size
        return top5_pos, top5_neg

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

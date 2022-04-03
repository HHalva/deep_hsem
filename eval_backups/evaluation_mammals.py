import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

import pdb
import random
import shutil
import time
import warnings
import itertools
import pickle

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

parser = argparse.ArgumentParser(description='ImageNet ZSL evaluation')
parser.add_argument('--euclidean', default='False', action='store_true',
                    help='evaluate Euclidean rather than Poincare embedding')
parser.add_argument('-d', '--dim', default=50, type=int,
                    help='dimensionality of embedding')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='size of batches for evaluation')
parser.add_argument('--subgraph', default='all', type=str, metavar='G',
                    help='name of specifc subgraph as in wordnet closer \
                    (default: all uses full imagenet)')
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
    torch.manual_seed(10)


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

    # create target index --> class label dictionary
    idx_to_class_map = {v:k for k,v in eval_dataset.class_to_idx.items()}

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

    # compute test labels distance from training labels

    fname = 'dists_from_training_labels.pkl'

    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            test_to_training_dists = pickle.load(f)

    elif not os.path.isfile(fname):
        training_labels = [lab for lab in
                os.listdir("/mnt/fast-data15/datasets/ILSVRC/2012/clsloc/train/")]
        training_synsets = [wn.of2ss(i[1:]+'-n') for i in training_labels]

        def dist_to_train_lab(test_offset,  training_synsets):
            test_synset = wn.of2ss(test_offset[1:]+'-n')
            return min([test_synset.shortest_path_distance(i) for
                         i in training_synsets])

        test_to_training_dists = {k: dist_to_train_lab(v, training_synsets)
                                  for k, v in idx_to_class_map.items()}
        # save results
        with open(fname, 'wb') as f:
            pickle.dump(test_to_training_dists, f, pickle.HIGHEST_PROTOCOL)


    #trackers
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':.3f')
    top2 = AverageMeter('Acc@2', ':.3f')
    top5 = AverageMeter('Acc@5', ':.3f')
    top10 = AverageMeter('Acc@10', ':.3f')
    top20 = AverageMeter('Acc@20', ':.3f')

    progress = ProgressMeter(
        len(eval_loader),
        [batch_time, top1, top2, top5, top10, top20],
        prefix='Eval: ')

    cond_accs_dict = {'top-1': [], 'top-2': [], 'top-5': [],
                      'top-10': [], 'top-20': []}
    leaf_dists_dict = {'top-1': [], 'top-2': [], 'top-5': [],
                       'top-10': [], 'top-20': []}
    hop_dists_dict = {'top-1': [], 'top-2': [], 'top-5': [],
                       'top-10': [], 'top-20': []}

    #evaluate
    model.eval()

    # rememeber to update below list with chosen subgraphs 
    assert args.subgraph in ['all', 'mammal', 'vehicle']

    if not args.subgraph == 'all':
        subgraph_ss = wn.synsets(args.subgraph, 'n')[0]
    with torch.no_grad():
        end = time.time()
        counter = 0
        for i, (images, target) in enumerate(eval_loader):
            counter += 1
            target_ss = wn.of2ss(idx_to_class_map[target.item()][1:]+'-n')

            # filter out all but relevant subgraph
            if not args.subgraph == 'all':
                hypers = lambda s: s.hypernyms()
                target_hypers = list(target_ss.closure(hypers))
                if not subgraph_ss in target_hypers:
                    continue

            # filter out all but leaf nodes
            if len(target_ss.hyponyms()) > 0:
                continue

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)

            # compute tree evaluations on wordnet
            cond_accs, leaf_dists = wn_eval(output, emb_wgts, target_ss,
                                            idx_to_class_map,
                                            (1, 2, 5, 10, 20), emb_name)
            # compute top-k accuracies
            (prec1, prec2, prec5, prec10, prec20), preds = accuracy(
                    output, emb_wgts, target, (1, 2, 5, 10, 20), emb_name)

            # track the distance of test labels from training labels
            hop_dists_dict['top-1'].extend([test_to_training_dists[target.item()]])
            hop_dists_dict['top-2'].extend([test_to_training_dists[target.item()]]*2)
            hop_dists_dict['top-5'].extend([test_to_training_dists[target.item()]]*5)
            hop_dists_dict['top-10'].extend([test_to_training_dists[target.item()]]*10)
            hop_dists_dict['top-20'].extend([test_to_training_dists[target.item()]]*20)

            # keep track of results
            cond_accs_dict['top-1'].extend(cond_accs[0])
            cond_accs_dict['top-2'].extend(cond_accs[1])
            cond_accs_dict['top-5'].extend(cond_accs[2])
            cond_accs_dict['top-10'].extend(cond_accs[3])
            cond_accs_dict['top-20'].extend(cond_accs[4])

            leaf_dists_dict['top-1'].extend(leaf_dists[0])
            leaf_dists_dict['top-2'].extend(leaf_dists[1])
            leaf_dists_dict['top-5'].extend(leaf_dists[2])
            leaf_dists_dict['top-10'].extend(leaf_dists[3])
            leaf_dists_dict['top-20'].extend(leaf_dists[4])

            # track accuracies
            top1.update(prec1.item(), images.size(0))
            top2.update(prec2.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))
            top10.update(prec10.item(), images.size(0))
            top20.update(prec20.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print evaluation 
            if i % args.print_freq == 0:
                progress.display(i)

        # save evaluation
        fprefix = str(args.dim)+emb_name[0]+"_"+args.subgraph
        with open(fprefix+"_cond_accs.txt", "wb") as fp:
            pickle.dump(cond_accs_dict, fp)

        with open(fprefix+"_leaf_dists.txt", "wb") as fp:
            pickle.dump(leaf_dists_dict, fp)

        with open(fprefix+"_hop_dists.txt", "wb") as fp:
            pickle.dump(hop_dists_dict, fp)

        # print final accuracies
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


def accuracy(output, all_embs, targets, topk, emb_type):
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


def wn_eval(output, all_embs, target_synset, idx_to_class_dictionary,
            topk=(1,), emb_type='poincare'):
    """Computes the accuracy over the k top predictions for the specified
    values of k"""
    with torch.no_grad():
        maxk = max(topk)
        preds = prediction(output, all_embs, maxk, emb_type)
        preds_ss = [wn.of2ss(idx_to_class_dictionary[p.item()][1:]+'-n')
                    for p in preds.squeeze()]
        preds_tree_dists = [tree_eval(target_synset, p)
                            for p in preds_ss]
        cond_accs, leaf_dists = list(zip(*preds_tree_dists))
        cond_accs_list, leaf_dists_list = [], []
        for k in topk:
            cond_accs_list.append(list(cond_accs[:k]))
            leaf_dists_list.append(list(leaf_dists[:k]))
        return cond_accs_list, leaf_dists_list


def tree_eval(target_ss, prediction_ss):
    target_hypers = list(target_ss.closure(lambda s: s.hypernyms()))
    target_hypers.append(target_ss)
    if prediction_ss in target_hypers:
        cond_acc = 1
        dist_to_leaf = target_ss.shortest_path_distance(prediction_ss)
    elif prediction_ss not in target_hypers:
        cond_acc = 0
        prediction_hypos = prediction_ss.hyponyms()
        dist_to_leaf = 0
        leaf_found = len(prediction_hypos) == 0
        while not leaf_found:
            dist_to_leaf += 1
            ph = [p.hyponyms() for p in prediction_hypos]
            leaf_found = [] in ph
            prediction_hypos = list(itertools.chain(*ph))
    return (cond_acc, dist_to_leaf)


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

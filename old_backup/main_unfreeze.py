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
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from hype_funcs.lorentz import LorentzManifold




from custom_loss import PoincareXEntropyLoss
#from poincare_model import PoincareDistance

from nltk.corpus import wordnet as wn

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay',  default=0.1, type=float,
                    metavar='D', help='decay factor for LR')
parser.add_argument('--lr-decay-interval',  default=30, type=float,
                    metavar='N', help='decay lr every N epoch')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--unfreeze', dest='unfreeze', action='store_true',
                    help='unfreeze all the layers for training')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--emb_dir', default='/home/hermanni/poincare-embeddings/',
		    type=str, help='location where embeddings are stored') 
parser.add_argument('--emb_name', default=None, type=str, required=True,
	            help='name of the embedding file')


best_prec1 = 0


def main():
    global args, best_prec1, poinc_emb
    global imgnet_poinc_wgt, imgnet_poinc_labels
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')


    if args.emb_name is None:
        raise NameError('args.emb_name file not specified') 


    #load poincare embedding
    poinc_emb = torch.load(args.emb_dir+args.emb_name)
    print('EMBEDDING TYPE:', poinc_emb['manifold'])    
    n_emb_dims = poinc_emb['embeddings'].shape[1]
    print('NUM OF DIMENSIONS:', n_emb_dims)

    #change labels from synset names into imagenet format
    synset_list = [wn.synset(i) for i in poinc_emb['objects']]
    offset_list = [wn.ss2of(j) for j in synset_list]
    poinc_emb['objects'] = ['n'+i.split('-')[0] for i in offset_list]

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        orig_vgg = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        orig_vgg = models.__dict__[args.arch]()

    #Change model to project into poincare space
    model = PoincareVGG(orig_vgg, n_emb_dims, args.unfreeze)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = PoincareXEntropyLoss()
    if args.unfreeze:
        optimizer = torch.optim.SGD([{'params': model.features.parameters(),
                                      'lr': args.lr*10**-1},
                                     {'params': model.fc.parameters()},
                                     {'params': model.classifier.parameters()}],
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                           model.parameters()),
                                           args.lr,
                                           momentum=args.momentum,
                                           weight_decay=args.weight_decay)

    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer,
            step_size=args.lr_decay_interval, gamma=args.lr_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_sched.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val_white')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
                  traindir,
                  transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                  normalize,]))

    #create poincare embedding that only contains imagenet synsets
    imgnet_poinc_labels = train_dataset.classes
    imgnet2poinc_idx = [poinc_emb['objects'].index(i) 
                        for i in imgnet_poinc_labels]
    imgnet_poinc_wgt = poinc_emb['embeddings'][imgnet2poinc_idx]

    #create train and val data loaders 
    train_loader = torch.utils.data.DataLoader(
                 train_dataset, batch_size=args.batch_size,
                 shuffle=True, num_workers=args.workers,
                 pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
               datasets.ImageFolder(valdir, transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     normalize,])),
               batch_size=args.batch_size, shuffle=False,
               num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        lr_sched.step()

        # train the model 
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            'scheduler': lr_sched.state_dict(),
        }, is_best, args.emb_name+'_checkp.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    #needed for converting class idx to IDs
    class2idx = train_loader.dataset.class_to_idx
    idx2class = {v: k for k, v in class2idx.items()}

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)
        target_ids = [idx2class[i.item()] for i in target]
        target_emb_idx = [imgnet_poinc_labels.index(i) for i in target_ids]
        target_embs = imgnet_poinc_wgt[target_emb_idx]
        target_embs = target_embs.cuda(args.gpu, non_blocking=True)

        # compute output & loss
        output = model(input)
        pdb.set_trace()
        loss = criterion(output, target, imgnet_poinc_wgt)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, imgnet_poinc_wgt, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    #needed for converting class idx to IDs
    class2idx = val_loader.dataset.class_to_idx
    idx2class = {v: k for k, v in class2idx.items()}

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            target_ids = [idx2class[i.item()] for i in target]
            target_emb_idx = [imgnet_poinc_labels.index(i) for i in target_ids]
            target_embs = imgnet_poinc_wgt[[target_emb_idx]]
            target_embs = target_embs.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target, imgnet_poinc_wgt)

            # measure accuracy and record loss
            prec1, prec5  = accuracy(output, imgnet_poinc_wgt, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best_'+filename)


class PoincareVGG(nn.Module):
    def __init__(self, vgg_model, n_emb_dims, unfreeze=False):
        super(PoincareVGG, self).__init__()
        self.features = vgg_model.features
        self.fc = nn.Sequential(*list(
                                vgg_model.classifier.children())[:-1])
        self.classifier = nn.Sequential(nn.Linear(4096, n_emb_dims))
        self.dir_func = nn.Linear(n_emb_dims, n_emb_dims, bias=False)
        self.norm_func = nn.Linear(n_emb_dims, 1, bias=False)
        nn.init.uniform_(self.dir_func.weight, -0.001, 0.001)
        nn.init.uniform_(self.norm_func.weight, -0.001, 0.001)

        #default is to unfreeze classifier i.e. fully connected layers
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
        p = F.sigmoid(norms_magnitude)
        return p*v


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def prediction(output, all_embs, knn=1):
    """Predicts the nearest class based on poincare distance"""
    with torch.no_grad():
        batch_size = output.size(0)
        n_emb_dims = output.size(1)
        n_classes = all_embs.size(0)
        expand_output = output.repeat(1, n_classes).view(-1, n_emb_dims)
        expand_all_embs = all_embs.repeat(batch_size, 1)
        dists_to_all = PoincareDistance.apply(expand_output,
                                              expand_all_embs.cuda(args.gpu,
                                                  non_blocking=True))
        topk_per_batch = torch.topk(dists_to_all.view(batch_size, -1),
                                     k=knn, dim=1,
                                     largest=False)[1]
        if knn==1:
            return topk_per_batch.view(-1)
        return topk_per_batch


def accuracy(output, all_embs, targets, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        preds = prediction(output, all_embs, knn=maxk)
        batch_size = output.size(0)
        res = []
        for k in topk:
            i = k
            preds_tmp = preds[:, :i]
            correct_tmp = preds_tmp.eq(targets.view(batch_size, -1).repeat(1, i))
            res.append(torch.sum(correct_tmp).float() / batch_size)
        return res

if __name__ == '__main__':
    main()

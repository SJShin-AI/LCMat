import time, torch
from argparse import ArgumentTypeError
from prefetch_generator import BackgroundGenerator
import os
import csv
import torch

class WeightedSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, weights) -> None:
        self.dataset = dataset
        assert len(indices) == len(weights)
        self.indices = indices
        self.weights = weights

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]], self.weights[[i for i in idx]]
        return self.dataset[self.indices[idx]], self.weights[idx]

def set_exp_with_args(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.selection == 'worst_match':
        args.backpack = True

    if args.train_batch is None:
        args.train_batch = args.batch
    if args.selection_batch is None:
        args.selection_batch = args.batch
    if args.save_path != "" and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if args.resume != "":
        # Load checkpoint
        try:
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            assert {"exp", "epoch", "state_dict", "opt_dict", "best_acc1", "rec", "subset", "sel_args"} <= set(
                checkpoint.keys())
            assert 'indices' in checkpoint["subset"].keys()
            start_exp = checkpoint['exp']
            start_epoch = checkpoint["epoch"]
        except AssertionError:
            try:
                assert {"exp", "subset", "sel_args"} <= set(checkpoint.keys())
                assert 'indices' in checkpoint["subset"].keys()
                print("=> The checkpoint only contains the subset, training will start from the begining")
                start_exp = checkpoint['exp']
                start_epoch = 0
            except AssertionError:
                print("=> Failed to load the checkpoint, an empty one will be created")
                checkpoint = {}
                start_exp = 0
                start_epoch = 0
    else:
        checkpoint = {}
        start_exp = 0
        start_epoch = 0

    exp = args.seed
    if args.save_path != "":
        if args.val:
            checkpoint_name = "{dst}_{net}_{mtd}_exp{exp}_{fr}_val{val_ratio}_se{selection_epochs}_".format(
                dst=args.dataset,
                net=args.model,
                mtd=args.selection,
                exp=exp,
                fr=args.fraction,
                val_ratio=args.val_ratio,
                selection_epochs=args.selection_epochs)
        else:
            if args.selection == 'Uncertainty':
                checkpoint_name = "{dst}_{net}_{mtd}_exp{exp}_{fr}_se{selection_epochs}_unc{uncertainty}".format(
                    dst=args.dataset,
                    net=args.model,
                    mtd=args.selection,
                    exp=exp,
                    fr=args.fraction,
                    selection_epochs=args.selection_epochs,
                    uncertainty=args.uncertainty)
            elif args.selection == 'Submodular':
                if args.kernel == 'cosine':
                    checkpoint_name = "{dst}_{net}_{mtd}_exp{exp}_{fr}_se{selection_epochs}_{submodular}_{kernel}_".format(
                        dst=args.dataset,
                        net=args.model,
                        mtd=args.selection,
                        exp=exp,
                        fr=args.fraction,
                        selection_epochs=args.selection_epochs,
                        submodular=args.submodular,
                        kernel=args.kernel)
                elif 'worst' in args.kernel:
                    checkpoint_name = "{dst}_{net}_{mtd}_exp{exp}_{fr}_se{selection_epochs}_{submodular}_{kernel}_eps{eps}_K{k}_".format(
                        dst=args.dataset,
                        net=args.model,
                        mtd=args.selection,
                        exp=exp,
                        fr=args.fraction,
                        selection_epochs=args.selection_epochs,
                        submodular=args.submodular,
                        kernel=args.kernel,
                        eps=args.eps,
                        k=args.K)

                else:
                    checkpoint_name = "{dst}_{net}_{mtd}_exp{exp}_{fr}_se{selection_epochs}_{submodular}_{kernel}_".format(
                        dst=args.dataset,
                        net=args.model,
                        mtd=args.selection,
                        exp=exp,
                        fr=args.fraction,
                        selection_epochs=args.selection_epochs,
                        submodular=args.submodular,
                        kernel=args.kernel)
            else:
                checkpoint_name = "{dst}_{net}_{mtd}_exp{exp}_{fr}_se{selection_epochs}_".format(dst=args.dataset,
                                                                                                 net=args.model,
                                                                                                 mtd=args.selection,
                                                                                                 exp=exp,
                                                                                                 fr=args.fraction,
                                                                                                 selection_epochs=args.selection_epochs)

        checkpoint_name += 'selection_decay{w_d}_{opt}_'.format(w_d=args.selection_weight_decay,
                                                                opt=args.selection_optimizer)
        print('\n================== Exp %d ==================\n' % exp)
        print("dataset: ", args.dataset, ", model: ", args.model, ", selection: ", args.selection, ", num_ex: ",
              args.num_exp, ", epochs: ", args.epochs, ", fraction: ", args.fraction, ", seed: ", args.seed,
              ", lr: ", args.lr, ", save_path: ", args.save_path, ", resume: ", args.resume, ", device: ", args.device,
              ", checkpoint_name: " + checkpoint_name if args.save_path != "" else "", "\n", sep="")

        args.checkpoint_name = checkpoint_name
        args.model_name = '{dataset}_{model}_decay{weight_decay}_epoch{num_epochs}_{optimizer}_seed{seed}.pt'.format(dataset=args.dataset\
                         ,model=args.model, num_epochs=args.selection_epochs ,weight_decay=args.weight_decay,optimizer=args.selection_optimizer, seed=args.seed)

        print('\n================== Model : %s ==================\n' % args.model_name)

    return args, exp, checkpoint, checkpoint_name, start_exp, start_epoch


def train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted: bool = False):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to train mode
    network.train()

    end = time.time()
    for i, contents in enumerate(train_loader):
        optimizer.zero_grad()
        if if_weighted:
            target = contents[0][1].to(args.device)
            input = contents[0][0].to(args.device)

            # Compute output
            output = network(input)
            weights = contents[1].to(args.device).requires_grad_(False)
            loss = torch.sum(criterion(output, target) * weights) / torch.sum(weights)
        else:
            target = contents[1].to(args.device)
            input = contents[0].to(args.device)

            # Compute output
            output = network(input)
            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

    record_train_stats(rec, epoch, losses.avg, top1.avg, optimizer.state_dict()['param_groups'][0]['lr'])


def test(test_loader, network, criterion, epoch, args, rec):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.to(args.device)
        input = input.to(args.device)

        # Compute output
        with torch.no_grad():
            output = network(input)

            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    network.no_grad = False

    record_test_stats(rec, epoch, losses.avg, top1.avg)
    return top1.avg


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
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def str_to_bool(v):
    # Handle boolean type in arguments.
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def save_checkpoint(state, path, epoch, prec):
    print("=> Saving checkpoint for epoch %d, with Prec@1 %f." % (epoch, prec))
    torch.save(state, path)


def init_recorder():
    from types import SimpleNamespace
    rec = SimpleNamespace()
    rec.train_step = []
    rec.train_loss = []
    rec.train_acc = []
    rec.lr = []
    rec.test_step = []
    rec.test_loss = []
    rec.test_acc = []
    rec.ckpts = []
    return rec

def record_train_stats(rec, step, loss, acc, lr):
    rec.train_step.append(step)
    rec.train_loss.append(loss)
    rec.train_acc.append(acc)
    rec.lr.append(lr)
    return rec

def record_test_stats(rec, step, loss, acc):
    rec.test_step.append(step)
    rec.test_loss.append(loss)
    rec.test_acc.append(acc)
    return rec


def record_ckpt(rec, step):
    rec.ckpts.append(step)
    return rec


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def save_important_statistics(args, dict, name):
    os.makedirs(os.path.join(args.save_path,'csv'),exist_ok=True)
    with open(os.path.join(args.save_path, 'csv','Config_'+name +'_'+args.checkpoint_name +'.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        for key in dict:
            w.writerow([key,str(dict[key])])
    return


def save_dicts_for_analyses(args, dict):
    os.makedirs(os.path.join(args.save_path, 'results_analyses'), exist_ok=True)
    with open(os.path.join(args.save_path, 'results_analyses',args.model_name[:-3]+'_eps{eps}_frac{frac}'.format(eps=args.eps,frac=args.fraction)+'.pickle'), 'wb') as fw:
        pickle.dump(dict, fw, protocol=pickle.HIGHEST_PROTOCOL)

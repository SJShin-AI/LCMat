import os
import torch.nn as nn
import argparse
import deepcore.nets as nets
import deepcore.datasets as datasets
import deepcore.methods as methods
from torchvision import transforms
from utils import *
from datetime import datetime
from time import sleep
from collections import OrderedDict

def main():

    #################################################################################################

    parser = argparse.ArgumentParser(description='Parameter Processing')

    # Basic arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18', help='model')
    parser.add_argument('--selection', type=str, default="uniform", help="selection method")
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use')
    parser.add_argument('--print_freq', '-p', default=100, type=int, help='print frequency (default: 20)')
    parser.add_argument('--fraction', default=0.1, type=float, help='fraction of data to be selected (default: 0.1)')
    parser.add_argument('--seed', default=int(time.time() * 1000) % 100000, type=int, help="random seed")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--cross", type=str, nargs="+", default=None, help="models for cross-architecture experiments")

    # Optimizer and scheduler
    parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='minimum learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument("--nesterov", default=True, type=str_to_bool, help="if set nesterov")
    parser.add_argument("--scheduler", default="CosineAnnealingLR", type=str, help=
    "Learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for StepLR")
    parser.add_argument("--step_size", type=float, default=50, help="Step size for StepLR")

    # Training
    parser.add_argument('--batch', '--batch-size', "-b", default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument("--train_batch", "-tb", default=None, type=int,
                     help="batch size for training, if not specified, it will equal to batch size in argument --batch")
    parser.add_argument("--selection_batch", "-sb", default=None, type=int,
                     help="batch size for selection, if not specified, it will equal to batch size in argument --batch")

    # Testing
    parser.add_argument("--test_interval", '-ti', default=1, type=int, help=
    "the number of training epochs to be preformed between two test epochs; a value of 0 means no test will be run (default: 1)")
    parser.add_argument("--test_fraction", '-tf', type=float, default=1.,
                        help="proportion of test dataset used for evaluating the model (default: 1.)")

    # Selecting
    parser.add_argument("--selection_epochs", "-se", default=40, type=int,
                        help="number of epochs whiling performing selection on full dataset")
    parser.add_argument('--selection_momentum', '-sm', default=0.9, type=float, metavar='M',
                        help='momentum whiling performing selection (default: 0.9)')
    parser.add_argument('--selection_weight_decay', '-swd', default=5e-4, type=float,
                        metavar='W', help='weight decay whiling performing selection (default: 5e-4)',
                        dest='selection_weight_decay')
    parser.add_argument('--selection_optimizer', "-so", default="SGD",
                        help='optimizer to use whiling performing selection, e.g. SGD, Adam')
    parser.add_argument("--selection_nesterov", "-sn", default=True, type=str_to_bool,
                        help="if set nesterov whiling performing selection")
    parser.add_argument('--selection_lr', '-slr', type=float, default=0.1, help='learning rate for selection')
    parser.add_argument("--selection_test_interval", '-sti', default=1, type=int, help=
    "the number of training epochs to be preformed between two test epochs during selection (default: 1)")
    parser.add_argument("--selection_test_fraction", '-stf', type=float, default=1.,
             help="proportion of test dataset used for evaluating the model while preforming selection (default: 1.)")
    parser.add_argument('--balance', default=True, type=str_to_bool,
                        help="whether balance selection is performed per class")

    # Algorithm
    parser.add_argument('--submodular', default="GraphCut", help="specifiy submodular function to use")
    parser.add_argument('--submodular_greedy', default="LazyGreedy", help="specifiy greedy algorithm for submodular optimization")
    parser.add_argument('--uncertainty', default="Entropy", help="specifiy uncertanty score to use")

    # Checkpoint and resumption
    parser.add_argument('--save_path', "-sp", type=str, default='', help='path to save results (default: do not save)')
    parser.add_argument('--resume', '-r', type=str, default='', help="path to latest checkpoint (default: do not load)")

    # New thang
    parser.add_argument('--val', type=str_to_bool, default=False, help='path to save results (default: do not save)')
    parser.add_argument('--val_ratio',  type=float, default=0.2, help="path to latest checkpoint (default: do not load)")
    parser.add_argument('--backpack',  type=str_to_bool, default=False, help='whether to utilize backpack or not')

    # submodular new thang
    parser.add_argument('--eps',  type=float, default=0.05, help="path to latest checkpoint (default: do not load)")
    parser.add_argument('--kernel', type=str, default='cosine', help="path to latest checkpoint (default: do not load)")
    parser.add_argument('--K', type=int, default=100, help='how many sub-dimensions to choose from the whole parameter dimension')
    parser.add_argument('--after_analyses', type=str_to_bool, default=False, help='whether to utilize backpack or not')
    parser.add_argument('--exact_analyses', type=str_to_bool, default=False, help='whether to utilize backpack or not')

    #################################################################################################

    args = parser.parse_args()

    args, args.exp, args.checkpoint, args.checkpoint_name, start_exp, start_epoch = set_exp_with_args(args)

    # Dataset setting
    if args.val:
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test = datasets.__dict__[args.dataset] \
            (args.data_path,args)
    else:
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset] \
            (args.data_path)

    args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

    # Seed setting
    if args.seed is not None:
        import random
        torch.random.manual_seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # Subset setting
    if "subset" in args.checkpoint.keys():
        subset = args.checkpoint['subset']
        selection_args = args.checkpoint["sel_args"]
    else:
        selection_args = dict(epochs=args.selection_epochs,
                              selection_method=args.uncertainty,
                              balance=args.balance,
                              greedy=args.submodular_greedy,
                              function=args.submodular)
        # Method setting
        if args.val:
            method = methods.__dict__[args.selection](dst_train, args, args.fraction, args.seed,dst_val = dst_val,**selection_args)
        else:
            method = methods.__dict__[args.selection](dst_train, args, args.fraction, args.seed, **selection_args)
        subset = method.select()

    # Augmentation
    if args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
        dst_train.transform = transforms.Compose(
            [transforms.RandomCrop(args.im_size, padding=4, padding_mode="reflect"),
             transforms.RandomHorizontalFlip(), dst_train.transform])
    elif args.dataset == "ImageNet":
        dst_train.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    # Handle weighted subset
    if_weighted = "weights" in subset.keys()
    if if_weighted:
        dst_subset = WeightedSubset(dst_train, subset["indices"], subset["weights"])
    else:
        dst_subset = torch.utils.data.Subset(dst_train, subset["indices"])

    train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.train_batch, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.train_batch, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)


    # Listing cross-architecture experiment settings if specified.
    models = [args.model]
    if isinstance(args.cross, list):
        for model in args.cross:
            if model != args.model:
                models.append(model)

    test_dict = OrderedDict()
    test_dict['epochs'] = []
    test_dict['test_acc'] = []
    for model in models:
        if len(models) > 1:
            print("| Training on model %s" % model)

        network = nets.__dict__[model](channel, num_classes, im_size).to(args.device)

        if args.device == "cpu":
            print("Using CPU.")
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu[0])
            network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)
        elif torch.cuda.device_count() > 1:
            network = nets.nets_utils.MyDataParallel(network).cuda()

        if "state_dict" in args.checkpoint.keys():
            # Loading model state_dict
            network.load_state_dict(args.checkpoint["state_dict"])

        criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)

        # Optimizer
        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer == "Adam":
            optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum,
                                                             weight_decay=args.weight_decay, nesterov=args.nesterov)

        # LR scheduler
        if args.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs,
                                                                   eta_min=args.min_lr)
        elif args.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * args.step_size,
                                                        gamma=args.gamma)
        else:
            scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)
        scheduler.last_epoch = (start_epoch - 1) * len(train_loader)

        if "opt_dict" in args.checkpoint.keys():
            optimizer.load_state_dict(args.checkpoint["opt_dict"])

        # Log recorder
        if "rec" in args.checkpoint.keys():
            rec = args.checkpoint["rec"]
        else:
            rec = init_recorder()

        best_prec1 = args.checkpoint["best_acc1"] if "best_acc1" in args.checkpoint.keys() else 0.0

        # Save the checkpont with only the susbet.
        if args.save_path != "" and args.resume == "":
            save_checkpoint({"exp": args.exp,
                             "subset": subset,
                             "sel_args": selection_args},
                            os.path.join(args.save_path, args.checkpoint_name + ("" if model == args.model else model
                                         + "_") + "unknown.ckpt"), 0, 0.)

        for epoch in range(start_epoch, args.epochs):
            # train for one epoch
            train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=if_weighted)

            # evaluate on validation set
            if args.test_interval > 0 and (epoch + 1) % args.test_interval == 0:
                prec1 = test(test_loader, network, criterion, epoch, args, rec)
                test_dict['epochs'].append(epoch)
                test_dict['test_acc'].append(prec1)

                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1

                if is_best:
                    best_prec1 = prec1
                    if args.save_path != "":
                        rec = record_ckpt(rec, epoch)
                        save_checkpoint({"exp": args.exp,
                                         "epoch": epoch + 1,
                                         "best_acc1": best_prec1,
                                         "subset": subset,
                                         "sel_args": selection_args},
                                        os.path.join(args.save_path, args.checkpoint_name + (
                                            "" if model == args.model else model + "_") + "unknown.ckpt"),
                                        epoch=epoch, prec=best_prec1)

        # Prepare for the next checkpoint
        if args.save_path != "":
            try:
                os.rename(
                    os.path.join(args.save_path, args.checkpoint_name + ("" if model == args.model else model + "_") +
                                 "unknown.ckpt"), os.path.join(args.save_path, args.checkpoint_name +
                                 ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1))
            except:
                save_checkpoint({"exp": args.exp,
                                 "epoch": args.epochs,
                                 "best_acc1": best_prec1,
                                 "subset": subset,
                                 "sel_args": selection_args},
                                os.path.join(args.save_path, args.checkpoint_name +
                                             ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1),
                                epoch=args.epochs - 1,
                                prec=best_prec1)

        print('| Best accuracy: ', best_prec1, ", on model " + model if len(models) > 1 else "", end="\n\n")
        # save_important_statistics(args, test_dict, 'test_acc')
        start_epoch = 0
        args.checkpoint = {}
        sleep(2)


if __name__ == '__main__':
    main()

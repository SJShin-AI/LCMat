from .coresetmethod import CoresetMethod
import torch, time
from torch import nn
import numpy as np
from copy import deepcopy
from .. import nets
from torchvision import transforms
from utils import *
from backpack import backpack, extend
from backpack.extensions import BatchGrad, DiagHessian, BatchDiagHessian
from collections import OrderedDict
import os
from .methods_utils import *
from torch.autograd.functional import hessian


class EarlyTrain(CoresetMethod):
    '''
    Core code for training related to coreset selection methods when pre-training is required.
    '''

    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None,
                 torchvision_pretrain: bool = False, dst_pretrain_dict: dict = {}, fraction_pretrain=1., dst_test=None,
                 **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)

        self.model_name = args.model_name
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model

        if fraction_pretrain <= 0. or fraction_pretrain > 1.:
            raise ValueError("Illegal pretrain fraction value.")
        self.fraction_pretrain = fraction_pretrain

        if dst_pretrain_dict.__len__() != 0:
            dict_keys = dst_pretrain_dict.keys()
            if 'im_size' not in dict_keys or 'channel' not in dict_keys or 'dst_train' not in dict_keys or \
                    'num_classes' not in dict_keys:
                raise AttributeError(
                    'Argument dst_pretrain_dict must contain imszie, channel, dst_train and num_classes.')
            if dst_pretrain_dict['im_size'][0] != args.im_size[0] or dst_pretrain_dict['im_size'][0] != args.im_size[0]:
                raise ValueError("im_size of pretrain dataset does not match that of the training dataset.")
            if dst_pretrain_dict['channel'] != args.channel:
                raise ValueError("channel of pretrain dataset does not match that of the training dataset.")
            if dst_pretrain_dict['num_classes'] != args.num_classes:
                self.num_classes_mismatch()

        self.dst_pretrain_dict = dst_pretrain_dict
        self.torchvision_pretrain = torchvision_pretrain
        self.if_dst_pretrain = (len(self.dst_pretrain_dict) != 0)

        if torchvision_pretrain:
            # Pretrained models in torchvision only accept 224*224 inputs, therefore we resize current
            # datasets to 224*224.
            if args.im_size[0] != 224 or args.im_size[1] != 224:
                self.dst_train = deepcopy(dst_train)
                self.dst_train.transform = transforms.Compose([self.dst_train.transform, transforms.Resize(224)])
                if self.if_dst_pretrain:
                    self.dst_pretrain_dict['dst_train'] = deepcopy(dst_pretrain_dict['dst_train'])
                    self.dst_pretrain_dict['dst_train'].transform = transforms.Compose(
                        [self.dst_pretrain_dict['dst_train'].transform, transforms.Resize(224)])
        if self.if_dst_pretrain:
            self.n_pretrain = len(self.dst_pretrain_dict['dst_train'])
        self.n_pretrain_size = round(
            self.fraction_pretrain * (self.n_pretrain if self.if_dst_pretrain else self.n_train))
        self.dst_test = dst_test

    def train(self, epoch, list_of_train_idx, **kwargs):
        """ Train model for one epoch """

        self.before_train()
        self.model.train()

        print('\n=> Training Epoch #%d' % epoch)
        trainset_permutation_inds = np.random.permutation(list_of_train_idx)
        batch_sampler = torch.utils.data.BatchSampler(trainset_permutation_inds, batch_size=self.args.selection_batch,
                                                      drop_last=False)
        trainset_permutation_inds = list(batch_sampler)

        train_loader = torch.utils.data.DataLoader(self.dst_pretrain_dict['dst_train'] if self.if_dst_pretrain
                                                   else self.dst_train, shuffle=False, batch_sampler=batch_sampler,
                                                   num_workers=self.args.workers, pin_memory=True)

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            # Forward propagation, compute loss, get predictions
            self.model_optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.after_loss(outputs, loss, targets, trainset_permutation_inds[i], epoch)

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()

            self.while_update(outputs, loss, targets, epoch, i, self.args.selection_batch)

            loss.backward()
            self.model_optimizer.step()
        return self.finish_train()

    def run(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.train_indx = np.arange(self.n_train)

        # Setup model and loss
        self.model = nets.__dict__[self.args.model if self.specific_model is None else self.specific_model](
            self.args.channel, self.dst_pretrain_dict["num_classes"] if self.if_dst_pretrain else self.num_classes,
            pretrained=self.torchvision_pretrain,
            im_size=(224, 224) if self.torchvision_pretrain else self.args.im_size,backpack =self.args.backpack).to(self.args.device)

        if self.args.device == "cpu":
            print("Using CPU.")
        elif self.args.gpu is not None:
            torch.cuda.set_device(self.args.gpu[0])
            self.model = nets.nets_utils.MyDataParallel(self.model, device_ids=self.args.gpu)
        elif torch.cuda.device_count() > 1:
            self.model = nets.nets_utils.MyDataParallel(self.model).cuda()

        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.criterion.__init__()

        # Setup optimizer
        if self.args.selection_optimizer == "SGD":
            self.model_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.selection_lr,
                                                   momentum=self.args.selection_momentum,
                                                   weight_decay=self.args.selection_weight_decay,
                                                   nesterov=self.args.selection_nesterov)
        elif self.args.selection_optimizer == "Adam":
            self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.selection_lr,
                                                    weight_decay=self.args.selection_weight_decay)
        else:
            self.model_optimizer = torch.optim.__dict__[self.args.selection_optimizer](self.model.parameters(),
                                                                       lr=self.args.selection_lr,
                                                                       momentum=self.args.selection_momentum,
                                                                       weight_decay=self.args.selection_weight_decay,
                                                                       nesterov=self.args.selection_nesterov)

        self.before_run()

        os.makedirs('././pretrained',exist_ok=True)
        print('\n================== Start Model Learning ==================\n')
        if os.path.exists('././pretrained/'+self.model_name):
            print('\nLoading the pre-trained model of {model_name}'.format(model_name = self.model_name))
            print('././pretrained/'+self.model_name)
            self.model.load_state_dict(torch.load('././pretrained/'+self.model_name))

        else:
            print('\nLearning new model of {model_name}'.format(model_name = self.model_name))
            for epoch in range(self.epochs):
                list_of_train_idx = np.random.choice(np.arange(self.n_pretrain if self.if_dst_pretrain else self.n_train),
                                                     self.n_pretrain_size, replace=False)
                self.before_epoch()
                self.train(epoch, list_of_train_idx)
                if self.dst_test is not None and self.args.selection_test_interval > 0 and (epoch + 1) % self.args.selection_test_interval == 0:
                    self.test(epoch)
                self.after_epoch()
            torch.save(self.model.state_dict(), '././pretrained/'+self.model_name)

        return self.finish_run()

    def test(self, epoch):

        self.model.no_grad = True
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(self.dst_test if self.args.selection_test_fraction == 1. else
                                                  torch.utils.data.Subset(self.dst_test, np.random.choice(
                                                      np.arange(len(self.dst_test)),
                                                      round(len(self.dst_test) * self.args.selection_test_fraction),
                                                      replace=False)),
                                                  batch_size=self.args.selection_batch, shuffle=False,
                                                  num_workers=self.args.workers, pin_memory=True)
        correct = 0.
        total = 0.

        print('\n=> Testing Epoch #%d' % epoch)

        for batch_idx, (input, target) in enumerate(test_loader):
            output = self.model(input.to(self.args.device))
            loss = self.criterion(output, target.to(self.args.device)).sum()

            predicted = torch.max(output.data, 1).indices.cpu()
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % self.args.print_freq == 0:
                print('| Test Epoch [%3d/%3d] Iter[%3d/%3d]\t\tTest Loss: %.4f Test Acc: %.3f%%' % (
                    epoch, self.epochs, batch_idx + 1, (round(len(self.dst_test) * self.args.selection_test_fraction) //
                                                        self.args.selection_batch) + 1, loss.item(),
                    100. * correct / total))

        self.model.no_grad = False

    def num_classes_mismatch(self):
        pass

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        pass

    def finish_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def finish_run(self):
        pass

    def select(self, **kwargs):

        selection_result = self.run()

        self.selection_result  =selection_result
        return selection_result

    def cal_loss_gradient_eigen(self, index=None):
        subset = self.selection_result
        if_weighted = "weights" in subset.keys()

        if index is None:
            temporal_index = subset["indices"]
            pass
        else:
            # print(subset["indices"])
            # print(index)
            # print(subset["indices"].shape)
            # print(index.shape)
            # print(np.intersect1d(subset["indices"],index))
            # print(np.intersect1d(subset["indices"],index).shape)
            temporal_index, subset_selected_index, _ = np.intersect1d(subset["indices"],index,return_indices=True)

        if if_weighted:
            if index is None:
                self.dst_subset = WeightedSubset(self.dst_train, temporal_index, subset["weights"])
            else:
                self.dst_subset = WeightedSubset(self.dst_train, temporal_index, subset["weights"][subset_selected_index])
        else:
            self.dst_subset = torch.utils.data.Subset(self.dst_train, temporal_index)

        try:
            self.criterion = extend(self.criterion)
        except:
            pass

        original_batch_loader = torch.utils.data.DataLoader(
            self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
            batch_size=self.args.selection_batch,
            num_workers=self.args.workers)

        subset_batch_loader = torch.utils.data.DataLoader(
            self.dst_subset,
            batch_size=self.args.selection_batch,
            num_workers=self.args.workers)

        sample_num = self.n_train if index is None else len(index)
        self.embedding_dim = self.model.get_last_layer().in_features

        # Initialize a matrix to save gradients. (on cpu)
        losses = []
        gradients = []
        hessians = []

        for i, contents in enumerate(original_batch_loader):
            self.model_optimizer.zero_grad()
            if if_weighted:
                # print(contents)
                # print(contents)
                targets = contents[1]
                input = contents[0]
            else:
                targets = contents[1]
                input = contents[0]

            outputs = self.model(input.to(self.args.device))
            loss = self.criterion(outputs,
                                  targets.to(self.args.device))
            # batch_num = targets.shape[0]

            with backpack(BatchGrad(), BatchDiagHessian()):
                loss.backward()

            for name, param in self.model.named_parameters():
                if 'linear.weight' in name:
                    weight_parameters_grads = param.grad_batch
                    weight_parameters_hesses = param.diag_h_batch
                elif 'linear.bias' in name:
                    bias_parameters_grads = param.grad_batch
                    bias_parameters_hesses = param.diag_h_batch

            losses.append(loss.detach().cpu().numpy())
            gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
                                       dim=1).cpu().numpy())

            hessians.append(torch.cat([bias_parameters_hesses, weight_parameters_hesses.flatten(1)],
                                      dim=1).cpu().numpy())

        losses = np.array(losses)
        gradients = np.concatenate(gradients, axis=0)
        hessians = np.concatenate(hessians, axis=0)


        # Initialize a matrix to save gradients. (on cpu)\
        losses_subset = []
        gradients_subset = []
        hessians_subset = []

        for i, (contents, index) in enumerate(subset_batch_loader):

            if if_weighted:
                input = contents[0]
                targets = contents[1]
            else:
                input = contents
                targets = index
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device))
            loss = self.criterion(outputs,
                                  targets.to(self.args.device))
            batch_num = targets.shape[0]

            with backpack(BatchGrad(), BatchDiagHessian()):
                loss.backward()

            for name, param in self.model.named_parameters():
                if 'linear.weight' in name:
                    weight_parameters_grads = param.grad_batch
                    weight_parameters_hesses = param.diag_h_batch
                elif 'linear.bias' in name:
                    bias_parameters_grads = param.grad_batch
                    bias_parameters_hesses = param.diag_h_batch

            losses_subset.append(loss.detach().cpu().numpy())
            gradients_subset.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
                                       dim=1).cpu().numpy())
            hessians_subset.append(torch.cat([bias_parameters_hesses, weight_parameters_hesses.flatten(1)],
                                      dim=1).cpu().numpy())

        losses_subset = np.array(losses_subset)
        gradients_subset = np.concatenate(gradients_subset, axis=0)
        hessians_subset = np.concatenate(hessians_subset, axis=0)

        loss_difference = np.abs(losses.mean() - losses_subset.mean())
        gradient_difference_norm = np.linalg.norm(gradients.mean(axis=0)-gradients_subset.mean(axis=0))
        hessian_difference_norm = np.linalg.norm(hessians.mean(axis=0)-hessians_subset.mean(axis=0),1)
        hessian_max_eigen = max_diff_np(hessians.mean(axis=0),hessians_subset.mean(axis=0))

        print('===========important statistics==========')
        print(loss_difference)
        print(gradient_difference_norm)
        print(hessian_difference_norm)
        print(hessian_max_eigen)

        return loss_difference, gradient_difference_norm, hessian_difference_norm, hessian_max_eigen


    def save_feature_and_classifier(self, index=None):
        eigenvalue_list = []
        subset = self.selection_result
        if_weighted = "weights" in subset.keys()

        if index is None:
            temporal_index = subset["indices"]
            pass
        else:
            temporal_index, subset_selected_index, _ = np.intersect1d(subset["indices"],index,return_indices=True)
        if if_weighted:
            if index is None:
                self.dst_subset = WeightedSubset(self.dst_train, temporal_index, subset["weights"])
            else:
                self.dst_subset = WeightedSubset(self.dst_train, temporal_index, subset["weights"][subset_selected_index])
        else:
            self.dst_subset = torch.utils.data.Subset(self.dst_train, temporal_index)

        original_batch_loader = torch.utils.data.DataLoader(
            self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
            batch_size=self.args.selection_batch,
            num_workers=self.args.workers)

        subset_batch_loader = torch.utils.data.DataLoader(
            self.dst_subset,
            batch_size=self.args.selection_batch,
            num_workers=self.args.workers)

        sample_num = self.n_train if index is None else len(index)
        self.embedding_dim = self.model.get_last_layer().in_features

        feats_original = []
        targets_original = []
        for i, (input, targets) in enumerate(original_batch_loader):
            self.model_optimizer.zero_grad()
            feats = self.model.forward_feat(input.to(self.args.device))
            feats_original.append(feats.detach().cpu().numpy())
            targets_original.append(targets.detach().cpu().numpy())
        feats_original = np.concatenate(feats_original, axis=0)
        targets_original = np.concatenate(targets_original, axis=0).flatten()

        feats_subset = []
        targets_subset = []
        for i, (contents, index) in enumerate(subset_batch_loader):
            if if_weighted:
                input = contents[0]
                targets = contents[1]
            else:
                input = contents
                targets = index

            self.model_optimizer.zero_grad()
            feats = self.model.forward_feat(input.to(self.args.device))
            feats_subset.append(feats.detach().cpu().numpy())
            targets_subset.append(targets.detach().cpu().numpy())
        feats_subset = np.concatenate(feats_subset, axis=0)
        targets_subset = np.concatenate(targets_subset, axis=0).flatten()

        classifier_weight = self.model.get_last_layer().weight.data.detach().cpu()
        classifier_bias = self.model.get_last_layer().bias.data.detach().cpu()

        def CrossEntropyLoss(output,target):
            output += 1e-7
            return -1 * torch.mean(torch.log(torch.exp(output)/torch.sum(torch.exp(output)))*target)

        def loss(x,y,w,b):
            logit = torch.mm(x.reshape(16,-1),w.T) + b.reshape(1,-1)
            return CrossEntropyLoss(logit,y)

        batch_num = 16

        def loss_subset(x,y,w,b):
            logit = torch.mm(x.reshape(2,-1),w.T) + b.reshape(1,-1)
            return CrossEntropyLoss(logit,y)

        batch_num_sub = 2

        targets_original_np = np.copy(targets_original)
        targets_original = torch.Tensor(targets_original)
        targets_original = targets_original.view(-1, 1).float()

        targets_subset_np = np.copy(targets_subset)
        targets_subset = torch.Tensor(targets_subset)
        targets_subset = targets_subset.view(-1, 1).float()

        #1 global
        origin_idx = np.arange(len(feats_original))

        num_samples = len(feats_original)
        num_iter = num_samples//batch_num
        # for feat, target in zip(feats_original,targets_original):
        for iter_idx in range(num_iter):
            feat, target = feats_original[iter_idx*batch_num:(iter_idx+1)*batch_num], targets_original[iter_idx*batch_num:(iter_idx+1)*batch_num]
            inputs = (torch.Tensor(feat).to(self.args.device), target.to(self.args.device), classifier_weight.to(self.args.device),classifier_bias.to(self.args.device))
            output = hessian(loss, inputs)
            hessian_for_w = torch.cat((output[2][2].reshape(5120, 5120), output[2][3].reshape(5120, 10)), 1)
            hessian_for_b = torch.cat((output[3][2].reshape(10, 5120), output[3][3].reshape(10, 10)), 1)
            hessian_for_weight = torch.cat((hessian_for_w,hessian_for_b),0)
            try:
                hessian_origin_updated += hessian_for_weight.detach()
            except:
                hessian_origin_updated = torch.zeros_like(hessian_for_weight)

            iter_idx += 1

        hessian_origin_updated /= len(feats_original)
        subset_idx = np.arange(len(feats_subset))
        num_samples = len(feats_subset)
        num_iter = num_samples//batch_num_sub

        for iter_idx in range(num_iter):
            feat, target = feats_subset[iter_idx * batch_num_sub:(iter_idx + 1) * batch_num_sub], targets_subset[iter_idx * batch_num_sub:(iter_idx + 1) * batch_num_sub]
            inputs = (torch.Tensor(feat).to(self.args.device), target.to(self.args.device), classifier_weight.to(self.args.device),classifier_bias.to(self.args.device))
            output = hessian(loss_subset, inputs)

            hessian_for_w = torch.cat((output[2][2].reshape(5120, 5120), output[2][3].reshape(5120, 10)), 1)
            hessian_for_b = torch.cat((output[3][2].reshape(10, 5120), output[3][3].reshape(10, 10)), 1)
            hessian_for_weight = torch.cat((hessian_for_w,hessian_for_b),0)
            try:
                hessian_subset_updated += hessian_for_weight.detach()
            except:
                hessian_subset_updated = torch.zeros_like(hessian_for_weight)

        hessian_subset_updated /= len(feats_subset)

        target_hessian = hessian_origin_updated - hessian_subset_updated
        eigenvalues= torch.linalg.eigvals(target_hessian.reshape(5130,5130)).real
        eigenvalue_list.append(torch.max(eigenvalues))

        #2 class-wise
        for c in range(self.num_classes):
            origin_c_idx = origin_idx[targets_original_np == c]
            subset_c_idx = subset_idx[targets_subset_np == c]

            feats_origin_temp = feats_original[origin_c_idx]
            target_origin_temp = targets_original[origin_c_idx]

            num_samples = len(feats_origin_temp)
            num_iter = num_samples // batch_num

            # for feat, target in zip(feats_origin_temp, target_origin_temp):
            for iter_idx in range(num_iter):
                feat, target = feats_origin_temp[iter_idx * batch_num:(iter_idx + 1) * batch_num], target_origin_temp[iter_idx * batch_num:(iter_idx + 1) * batch_num]
                inputs = (torch.Tensor(feat).to(self.args.device), target.to(self.args.device),
                          classifier_weight.to(self.args.device), classifier_bias.to(self.args.device))
                output = hessian(loss, inputs)
                # hessian_for_weight = output[2][2]
                hessian_for_w = torch.cat((output[2][2].reshape(5120, 5120), output[2][3].reshape(5120, 10)), 1)
                hessian_for_b = torch.cat((output[3][2].reshape(10, 5120), output[3][3].reshape(10, 10)), 1)
                hessian_for_weight = torch.cat((hessian_for_w, hessian_for_b), 0)

                try:
                    hessian_origin_updated += hessian_for_weight.detach()
                except:
                    hessian_origin_updated = torch.zeros_like(hessian_for_weight)

            hessian_origin_updated /= len(feats_origin_temp)

            feats_subset_temp = feats_subset[subset_c_idx]
            targets_subset_temp = targets_subset[subset_c_idx]
            num_samples = len(feats_subset_temp)
            num_iter = num_samples // batch_num_sub

            # for feat, target in zip(feats_subset_temp, targets_subset_temp):
            for iter_idx in range(num_iter):
                feat, target = feats_subset_temp[iter_idx * batch_num_sub:(iter_idx + 1) * batch_num_sub], targets_subset_temp[iter_idx * batch_num_sub:(iter_idx + 1) * batch_num_sub]
                inputs = (torch.Tensor(feat).to(self.args.device), target.to(self.args.device),
                          classifier_weight.to(self.args.device), classifier_bias.to(self.args.device))
                output = hessian(loss_subset, inputs)
                # hessian_for_weight = output[2][2]
                hessian_for_w = torch.cat((output[2][2].reshape(5120, 5120), output[2][3].reshape(5120, 10)), 1)
                hessian_for_b = torch.cat((output[3][2].reshape(10, 5120), output[3][3].reshape(10, 10)), 1)
                hessian_for_weight = torch.cat((hessian_for_w, hessian_for_b), 0)
                try:
                    hessian_subset_updated += hessian_for_weight.detach()
                except:
                    hessian_subset_updated = torch.zeros_like(hessian_for_weight)

            hessian_subset_updated /= len(feats_subset_temp)

            target_hessian = hessian_origin_updated - hessian_subset_updated

            # eigenvalues, _ = torch.eig(target_hessian.reshape(5130,5130))
            eigenvalues = torch.linalg.eigvals(target_hessian.reshape(5130, 5130)).real
            eigenvalue_list.append(torch.max(eigenvalues))

        return eigenvalue_list





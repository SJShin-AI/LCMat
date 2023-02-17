from .earlytrain import EarlyTrain
import numpy as np
import torch
from .methods_utils import *
from ..nets.nets_utils import MyDataParallel
from backpack import backpack, extend
from backpack.extensions import BatchGrad, DiagHessian, BatchDiagHessian
import os
import pickle
from torch import nn


def save_dicts_for_analyses(args, dict):
    os.makedirs(os.path.join(args.save_path, 'results_analyses'), exist_ok=True)
    with open(os.path.join(args.save_path, 'results_analyses',args.model_name[:-3]+'_eps{eps}_frac{frac}'.format(eps=args.eps,frac=args.fraction)+'.pickle'), 'wb') as fw:
        pickle.dump(dict, fw, protocol=pickle.HIGHEST_PROTOCOL)

class Submodular(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None, balance=False,
                 function="LogDeterminant", greedy="ApproximateLazyGreedy", metric="cossim", **kwargs):
        super(Submodular, self).__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)

        if greedy not in submodular_optimizer.optimizer_choices:
            raise ModuleNotFoundError("Greedy optimizer not found.")
        self._greedy = greedy
        self._metric = metric
        self._function = function
        self.value_dict = {}

        self.balance = balance
        self.criterion_for_loss = nn.CrossEntropyLoss(reduction='none').to(self.args.device)

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def calc_gradient(self, index=None):
        '''
        Calculate gradients matrix on current network for specified training dataset.
        '''
        self.model.eval()
        batch_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args.selection_batch,
                num_workers=self.args.workers)
        sample_num = self.n_train if index is None else len(index)

        self.embedding_dim = self.model.get_last_layer().in_features
        # Initialize a matrix to save gradients. (on cpu)
        gradients = []

        for i, (input, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device))
            loss = self.criterion(torch.nn.functional.softmax(outputs.requires_grad_(True), dim=1),
                                  targets.to(self.args.device)).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                weight_parameters_grads = self.model.embedding_recorder.embedding.view(batch_num, 1,
                                        self.embedding_dim).repeat(1, self.args.num_classes, 1) *\
                                        bias_parameters_grads.view(batch_num, self.args.num_classes,
                                        1).repeat(1, 1, self.embedding_dim)

                gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
                                            dim=1).cpu().numpy())

        gradients = np.concatenate(gradients, axis=0)
        return gradients

    def calc_gradient_and_hess(self, index=None):
        '''
        Calculate gradients matrix on current network for specified training dataset.
        '''
        self.criterion = extend(self.criterion)

        batch_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args.selection_batch,
                num_workers=self.args.workers)
        sample_num = self.n_train if index is None else len(index)
        self.embedding_dim = self.model.get_last_layer().in_features

        # Initialize a matrix to save gradients. (on cpu)
        gradients = []
        hessians = []

        for i, (input, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device))
            loss = self.criterion(outputs,targets.to(self.args.device))
            batch_num = targets.shape[0]

            with backpack(BatchGrad(),BatchDiagHessian()):
                loss.backward()

            for name, param in self.model.named_parameters():
                if 'linear.weight' in name or 'classifier.weight' in name:
                    weight_parameters_grads = param.grad_batch
                    weight_parameters_hesses = param.diag_h_batch
                    # print(weight_parameters_hesses.shape)
                elif 'linear.bias' in name or 'classifier.bias' in name:
                    bias_parameters_grads = param.grad_batch
                    bias_parameters_hesses = param.diag_h_batch
                    # print(bias_parameters_hesses.shape)

            gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
                                        dim=1).cpu().numpy())
            hessians.append(torch.cat([bias_parameters_hesses, weight_parameters_hesses.flatten(1)],
                                        dim=1).cpu().numpy())

        gradients = np.concatenate(gradients, axis=0)
        hessians = np.concatenate(hessians, axis=0)
        return gradients, hessians

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module
        # Turn on the embedding recorder and the no_grad flag
        with self.model.embedding_recorder:
            self.train_indx = np.arange(self.n_train)

            if self.balance:
                selection_result = np.array([], dtype=np.int64)
                for c in range(self.num_classes):
                    c_indx = self.train_indx[self.dst_train.targets == c]
                    # Calculate gradients into a matrix

                    if self.args.kernel == 'worst':
                        gradients, hessians = self.calc_gradient_and_hess(index=c_indx)
                        if self.args.exact_analyses:
                            self.value_dict['gradient_origin_' + str(c)] = gradients
                            self.value_dict['diag_hessian_origin_'+str(c)] = hessians
                            self.value_dict['index_origin_' + str(c)] = c_indx

                        hessians_reduced, pick_idx_var, var_statistics = hessian_pick_var(hessians, self.args.K)
                        submod_function = submodular_function.__dict__[self._function](index=c_indx,
                                                                                       similarity_kernel=lambda a,b: 10 - l2_norm_np(gradients[a], gradients[b])
                                                                                       - self.args.eps * l1_norm_np(hessians_reduced[a],hessians_reduced[b]))

                    elif self.args.kernel == 'adacore':
                        gradients, hessians = self.calc_gradient_and_hess(index=c_indx)

                        inverse_hessians = np.reciprocal(hessians)
                        precond_gradients = gradients * inverse_hessians
                        hessians_reduced, pick_idx_var, var_statistics = hessian_pick_var(hessians, self.args.K)
                        submod_function = submodular_function.__dict__[self._function](index=c_indx,
                                                                                       similarity_kernel=lambda a,b: 10 - l2_norm_np(precond_gradients[a], precond_gradients[b]))

                    elif self.args.kernel == 'grad_l2':
                        gradients, hessians = self.calc_gradient_and_hess(index=c_indx)
                        submod_function = submodular_function.__dict__[self._function](index=c_indx,
                                                                                       similarity_kernel=lambda a,
                                                                                                                b: 10 -l2_norm_np(gradients[a],gradients[b]))
                    else:
                        gradients = self.calc_gradient(index=c_indx)
                        submod_function = submodular_function.__dict__[self._function](index=c_indx,
                                            similarity_kernel=lambda a, b:cossim_np(gradients[a], gradients[b]))

                    # Instantiate a submodular function
                    submod_optimizer = submodular_optimizer.__dict__[self._greedy](args=self.args,
                                        index=c_indx, budget=round(self.fraction * len(c_indx)), already_selected=[])
                    c_selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                                 update_state=submod_function.update_state)
                    selection_result = np.append(selection_result, c_selection_result)
            else:
                # Calculate gradients into a matrix
                gradients = self.calc_gradient()

                # Instantiate a submodular function
                submod_function = submodular_function.__dict__[self._function](index=self.train_indx,
                                            similarity_kernel=lambda a, b: cossim_np(gradients[a], gradients[b]))
                submod_optimizer = submodular_optimizer.__dict__[self._greedy](args=self.args, index=self.train_indx,
                                                                                  budget=self.coreset_size)
                selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
                                                           update_state=submod_function.update_state)

            self.model.no_grad = False
        return {"indices": selection_result}



    def select(self, **kwargs):
        selection_result = self.run()
        self.selection_result = selection_result
        self.value_dict['index_subset']=self.selection_result

        return selection_result




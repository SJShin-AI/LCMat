from .earlytrain import EarlyTrain
import torch
import numpy as np
from .methods_utils import euclidean_dist
from ..nets.nets_utils import MyDataParallel
from utils import *
from backpack import backpack, extend
from backpack.extensions import BatchGrad, DiagHessian, BatchDiagHessian
from collections import OrderedDict
import os

class Herding(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200,
                 specific_model="ResNet18", balance: bool = False, metric="euclidean", **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs=epochs, specific_model=specific_model, **kwargs)

        if metric == "euclidean":
            self.metric = euclidean_dist
        elif callable(metric):
            self.metric = metric
        else:
            self.metric = euclidean_dist
            self.run = lambda: self.finish_run()

            def _construct_matrix(index=None):
                data_loader = torch.utils.data.DataLoader(
                    self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                    batch_size=self.n_train if index is None else len(index), num_workers=self.args.workers)
                inputs, _ = next(iter(data_loader))
                return inputs.flatten(1).requires_grad_(False).to(self.args.device)

            self.construct_matrix = _construct_matrix

        self.balance = balance

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def construct_matrix(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                sample_num = self.n_train if index is None else len(index)
                matrix = torch.zeros([sample_num, self.emb_dim], requires_grad=False).to(self.args.device)

                data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                            torch.utils.data.Subset(self.dst_train, index),
                                            batch_size=self.args.selection_batch,
                                            num_workers=self.args.workers)

                for i, (inputs, _) in enumerate(data_loader):
                    self.model(inputs.to(self.args.device))
                    matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = self.model.embedding_recorder.embedding

        self.model.no_grad = False
        return matrix

    def before_run(self):
        self.emb_dim = self.model.get_last_layer().in_features

    def herding(self, matrix, budget: int, index=None):

        sample_num = matrix.shape[0]

        if budget < 0:
            raise ValueError("Illegal budget size.")
        elif budget > sample_num:
            budget = sample_num

        indices = np.arange(sample_num)
        with torch.no_grad():
            mu = torch.mean(matrix, dim=0)
            select_result = np.zeros(sample_num, dtype=bool)

            for i in range(budget):
                if i % self.args.print_freq == 0:
                    print("| Selecting [%3d/%3d]" % (i + 1, budget))
                dist = self.metric(((i + 1) * mu - torch.sum(matrix[select_result], dim=0)).view(1, -1),
                                   matrix[~select_result])
                p = torch.argmax(dist).item()
                p = indices[~select_result][p]
                select_result[p] = True
        if index is None:
            index = indices
        return index[select_result]

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

        if self.balance:
            selection_result = np.array([], dtype=np.int32)
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]

                selection_result = np.append(selection_result, self.herding(self.construct_matrix(class_index),
                        budget=round(self.fraction * len(class_index)), index=class_index))
        else:
            selection_result = self.herding(self.construct_matrix(), budget=self.coreset_size)
        return {"indices": selection_result}

    def select_balance(self):
        """The same sampling proportions were used in each class separately."""
        np.random.seed(self.random_seed)
        self.index = np.array([], dtype=np.int64)
        all_index = np.arange(self.n_train)
        for c in range(self.num_classes):
            c_index = (self.dst_train.targets == c)
            self.index = np.append(self.index,
                                   np.random.choice(all_index[c_index], round(self.fraction * c_index.sum().item()),
                                                    replace=self.replace))
        return self.index

    def select(self, **kwargs):
        selection_result = self.run()
        self.selection_result  =selection_result

        if self.args.after_analyses:
            analyses_dict = OrderedDict()
            analyses_dict['checkpoint_name'] = self.args.checkpoint_name
            # eigen_dict = self.save_feature_and_classifier()

            '''difference for whole instances'''
            loss_difference, gradient_difference_norm, hessian_difference_norm, hessian_max_eigen = self.cal_loss_gradient_eigen()

            analyses_dict['global_loss_diff'] = loss_difference
            analyses_dict['global_grad_l2_norm'] = gradient_difference_norm
            analyses_dict['global_hess_l1_norm'] = hessian_difference_norm
            analyses_dict['global_hess_max_eigen'] = hessian_max_eigen
            # analyses_dict['global_hess_exact_max_eigen'] = eigen_dict[0]

            '''differences for class-wise instances'''
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                loss_difference, gradient_difference_norm, hessian_difference_norm, hessian_max_eigen = self.cal_loss_gradient_eigen(c_indx)
                analyses_dict['global_loss_diff_'+str(c)] = loss_difference
                analyses_dict['global_grad_l2_norm_'+str(c)] = gradient_difference_norm
                analyses_dict['global_hess_l1_norm_'+str(c)] = hessian_difference_norm
                # analyses_dict['global_hess_max_eigen_'+str(c)] = hessian_max_eigen

            save_important_statistics(self.args, analyses_dict, 'analyses')







        return selection_result


import numpy as np
from .coresetmethod import CoresetMethod
from backpack import backpack, extend
from backpack.extensions import BatchGrad, DiagHessian, BatchDiagHessian

class Uniform(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, balance=False, replace=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.balance = balance
        self.replace = replace
        self.n_train = len(dst_train)

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

    def select_no_balance(self):
        np.random.seed(self.random_seed)
        self.index = np.random.choice(np.arange(self.n_train), round(self.n_train * self.fraction),
                                      replace=self.replace)

        return  self.index

    def select(self, **kwargs):

        if self.args.after_analyses:
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                self.cal_loss_gradient_eigen(c_indx)

        return {"indices": self.select_balance() if self.balance else self.select_no_balance()}


    def cal_loss_gradient_eigen(self, index=None):
        subset = self.selection_result
        if_weighted = "weights" in subset.keys()
        if if_weighted:
            self.dst_subset = WeightedSubset(self.dst_train, subset["indices"], subset["weights"])
        else:
            self.dst_subset = torch.utils.data.Subset(self.dst_train, subset["indices"])

        try:
            self.criterion = extend(self.criterion)
        except:
            pass

        original_batch_loader = torch.utils.data.DataLoader(
            self.dst_train,
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

        for i, (input, targets) in enumerate(original_batch_loader):
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

            losses.append(loss.detach().cpu().numpy())
            gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
                                       dim=1).cpu().numpy())

            hessians.append(torch.cat([bias_parameters_hesses, weight_parameters_hesses.flatten(1)],
                                      dim=1).cpu().numpy())

        losses = np.concatenate(losses, axis=0)
        gradients = np.concatenate(gradients, axis=0)
        hessians = np.concatenate(hessians, axis=0)


        # Initialize a matrix to save gradients. (on cpu)\
        losses_subset = []
        gradients_subset = []
        hessians_subset = []

        for i, (input, targets) in enumerate(subset_batch_loader):
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

        losses_subset = np.concatenate(losses_subset, axis=0)
        gradients_subset = np.concatenate(gradients_subset, axis=0)
        hessians_subset = np.concatenate(hessians_subset, axis=0)

        loss_difference = np.abs(losses.mean() - losses_subset.mean())
        gradient_difference_norm = l2_norm_np(gradients.mean(axis=0),gradients_subset.mean(axis=0))
        hessian_difference_norm = l1_norm_np(hessians.mean(axis=0),hessians_subset.mean(axis=0))
        hessian_max_eigen = max_diff_np(hessians.mean(axis=0),hessians_subset.mean(axis=0))


        return loss_difference, gradient_difference_norm, hessian_difference_norm, hessian_max_eigen

#!/usr/bin/env python3

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from run_utils import build_all_matrices, train_test_split, SplitType, export, evaluate


class MF_MSE_PyTorch_model(torch.nn.Module):

    def __init__(self, n_users, n_items, n_factors):
        super(MF_MSE_PyTorch_model, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_factors = torch.nn.Embedding(num_embeddings = self.n_users, embedding_dim = self.n_factors)
        self.item_factors = torch.nn.Embedding(num_embeddings = self.n_items, embedding_dim = self.n_factors)
        self.layer_1 = torch.nn.Linear(in_features = self.n_factors, out_features = 1)
        self.activation_function = torch.nn.ReLU()

    def forward(self, user_coordinates, item_coordinates):
        current_user_factors = self.user_factors(user_coordinates)
        current_item_factors = self.item_factors(item_coordinates)
        prediction = torch.mul(current_user_factors, current_item_factors)
        prediction = self.layer_1(prediction)
        prediction = self.activation_function(prediction)
        return prediction

    def get_W(self):
        return self.user_factors.weight.detach().cpu().numpy()

    def get_H(self):
        return self.item_factors.weight.detach().cpu().numpy()


class DatasetIterator_URM(Dataset):

    def __init__(self, URM):
        URM = URM.tocoo()
        self.n_data_points = URM.nnz
        self.user_item_coordinates = np.empty((self.n_data_points, 2))
        self.user_item_coordinates[:,0] = URM.row.copy()
        self.user_item_coordinates[:,1] = URM.col.copy()
        self.rating = URM.data.copy().astype(np.float)
        self.user_item_coordinates = torch.Tensor(self.user_item_coordinates).type(torch.LongTensor)
        self.rating = torch.Tensor(self.rating)

    def __getitem__(self, index):
        """
        Format is (row, col, data)
        :param index:
        :return:
        """
        return self.user_item_coordinates[index, :], self.rating[index]

    def __len__(self):
        return self.n_data_points


class MFRecommender:

    def __init__(self):
        self.H = None
        self.W = None

    def fit(self, urm_train):
        n_factors = 100
        n_users, n_items = urm_train.shape
        pyTorchModel = MF_MSE_PyTorch_model(n_users, n_items, n_factors).to(device)
        lossFunction = torch.nn.MSELoss(size_average=False)
        learning_rate = 1e-4
        optimizer = torch.optim.Adagrad(pyTorchModel.parameters(), lr=learning_rate)
        batch_size = 200
        dataset_iterator = DatasetIterator_URM(urm_train)
        train_data_loader = DataLoader(dataset=dataset_iterator, batch_size=batch_size, shuffle=True, num_workers=2)
        for num_batch, (input_data, label) in enumerate(train_data_loader, 0):
            cumulative_loss = 0
            # On windows requires int64, on ubuntu int32
            # input_data_tensor = Variable(torch.from_numpy(np.asarray(input_data, dtype=np.int64))).to(self.device)
            input_data_tensor = Variable(input_data).to(device)
            label_tensor = Variable(label).to(device)
            user_coordinates = input_data_tensor[:, 0]
            item_coordinates = input_data_tensor[:, 1]
            # FORWARD pass
            prediction = pyTorchModel(user_coordinates, item_coordinates)
            # Pass prediction and label removing last empty dimension of prediction
            loss = lossFunction(prediction.view(-1), label_tensor)
            if num_batch % 100 == 0:
                print("Batch {} of {}, loss {:.4f}".format(num_batch, len(train_data_loader), loss.data.item()))
                if num_batch == 2000:
                    print("Interrupting train")
                    break
            # BACKWARD pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.W = pyTorchModel.get_W()
        self.H = pyTorchModel.get_H().T

    def recommend(self, user_id, at=None, exclude_seen=True):
        scores = np.dot(self.W[user_id, :], self.H)
        ranking = scores.argsort()[::-1]
        return ranking[:at]


if __name__ == '__main__':
    EXPORT = False
    urm, icm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.LOO)
    use_cuda = False
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("MF_MSE_PyTorch: Using CUDA")
    else:
        device = torch.device('cpu')
        print("MF_MSE_PyTorch: Using CPU")
    rec = MFRecommender()
    rec.fit(urm_train)
    if EXPORT:
        export(target_users, rec)
    else:
        evaluate(rec, urm_test)

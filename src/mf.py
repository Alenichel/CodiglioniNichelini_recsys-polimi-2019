#!/usr/bin/env python3

import numpy as np
import torch
import implicit
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from run_utils import build_all_matrices, train_test_split, SplitType, export, evaluate
from bayes_opt import BayesianOptimization

class MF_MSE_PyTorch_model(torch.nn.Module):

    def __init__(self, n_users, n_items, n_factors):
        super(MF_MSE_PyTorch_model, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_factors = torch.nn.Embedding(num_embeddings = self.n_users, embedding_dim = self.n_factors)
        self.item_factors = torch.nn.Embedding(num_embeddings = self.n_items, embedding_dim = self.n_factors)
        self.layer_1 = torch.nn.Linear(in_features = self.n_factors, out_features = 1)
        self.activation_function = torch.nn.Sigmoid()

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
        n_factors = 10
        n_users, n_items = urm_train.shape
        pyTorchModel = MF_MSE_PyTorch_model(n_users, n_items, n_factors).to(device)
        lossFunction = torch.nn.MSELoss(size_average=False)
        learning_rate = 1e-4
        optimizer = torch.optim.Adadelta(pyTorchModel.parameters(), lr=learning_rate)
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


class AlternatingLeastSquare:
    """
    ALS implemented with implicit following guideline of
    https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe
    IDEA:
    Recomputing x_{u} and y_i can be done with Stochastic Gradient Descent, but this is a non-convex optimization problem.
    We can convert it into a set of quadratic problems, by keeping either x_u or y_i fixed while optimizing the other.
    In that case, we can iteratively solve x and y by alternating between them until the algorithm converges.
    This is Alternating Least Squares.
    """

    def __init__(self):
        self.n_factors = None
        self.regularization = None
        self.iterations = None

    def fit(self, URM, n_factors=300, regularization=0.15, iterations=30):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.URM = URM
        sparse_item_user = self.URM.T

        # Initialize the als model and fit it using the sparse item-user matrix
        model = implicit.als.AlternatingLeastSquares(factors=self.n_factors, regularization=self.regularization, iterations=self.iterations)

        alpha_val = 24
        # Calculate the confidence by multiplying it by our alpha value.

        data_conf = (sparse_item_user * alpha_val).astype('double')

        # Fit the model
        model.fit(data_conf, show_progress=False)

        # Get the user and item vectors from our trained model
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors

    def filter_seen(self, user_id, scores):
        start_pos = self.urm.indptr[user_id]
        end_pos = self.urm.indptr[user_id + 1]
        user_profile = self.urm.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores

    def get_scores(self, user_id, exclude_seen=True):
        scores = np.dot(self.user_factors[user_id], self.item_factors.T)

        return np.squeeze(scores)

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_scores(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]


def tuner():
    urm, icm, ucm, _ = build_all_matrices()
    urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)
    pbounds = {'n_factors': (0, 1000), 'regularization': (0, 100), 'iterations': (0, 200)}

    def rec_round(n_factors, regularization, iterations):
        n_factors = int(n_factors)
        iterations = int(iterations)
        ALS = AlternatingLeastSquare()
        ALS.fit(urm_train, n_factors=n_factors, regularization=regularization, iterations=iterations)
        return evaluate(ALS, urm_test, cython=True, verbose=False)['MAP']

    optimizer = BayesianOptimization(f=rec_round, pbounds=pbounds)
    optimizer.maximize(init_points=10, n_iter=500)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)


if __name__ == '__main__':
    np.random.seed(42)
    """tuner()
    exit()"""

    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    ALS = AlternatingLeastSquare()
    ALS.fit(urm_train, n_factors=999, regularization=97.7, iterations=196)

    if EXPORT:
        export(target_users, ALS)
    else:
        evaluate(ALS, urm_test, cython=False, verbose=True)

from cbf import UserCBFKNNRecommender, ItemCBFKNNRecommender
from cf import ItemCFKNNRecommender, UserCFKNNRecommender
from clusterization import get_clusters, get_clusters_profile_length
from clusterized_top_pop import ClusterizedTopPop
from cython_modules.SLIM_BPR.SLIM_BPR_CYTHON import SLIM_BPR
from hybrid import HybridRecommender, MergingTechniques
from mf import AlternatingLeastSquare
from run_utils import build_all_matrices, user_segmenter, set_seed, train_test_split, SplitType, export, evaluate
from model_hybrid import ModelHybridRecommender
from slim_elasticnet import SLIMElasticNetRecommender
from basic_recommenders import TopPopRecommender


class GroupsHybridRecommender:

    def initiliaze_recommender(self):
        global urm_train, ucm
        # TOP-POP
        top_pop = TopPopRecommender()
        top_pop.fit(urm_train)
        # USER CBF
        user_cbf = UserCBFKNNRecommender()
        user_cbf.fit(urm_train, ucm, top_k=496, shrink=0, normalize=False)
        # HYBRID FALLBACK
        hybrid_fb = HybridRecommender([top_pop, user_cbf], urm_train, merging_type=MergingTechniques.MEDRANK)
        # ITEM CF
        item_cf = ItemCFKNNRecommender(fallback_recommender=hybrid_fb)
        item_cf.fit(urm_train, top_k=4, shrink=34, normalize=False, similarity='jaccard')
        # SLIM BPR
        slim_bpr = SLIM_BPR(fallback_recommender=hybrid_fb)
        slim_bpr.fit(urm_train, epochs=300)
        # SLIM ELASTICNET
        slim_enet = SLIMElasticNetRecommender(fallback_recommender=hybrid_fb)
        slim_enet.fit(urm_train, cache=not EXPORT)
        # MODEL HYBRID
        model_hybrid = ModelHybridRecommender([item_cf.w_sparse, slim_bpr.W, slim_enet.W_sparse], [42.82, 535.4, 52.17],
                                              fallback_recommender=hybrid_fb)
        model_hybrid.fit(urm_train, top_k=977)
        # USER CF
        user_cf = UserCFKNNRecommender()
        user_cf.fit(urm_train, top_k=593, shrink=4, normalize=False, similarity='tanimoto')
        # ITEM CBF
        item_cbf = ItemCBFKNNRecommender()
        item_cbf.fit(urm_train, icm, 417, 0.3, normalize=True)
        # ALS
        als = AlternatingLeastSquare()
        als.fit(urm_train, n_factors=896, regularization=99.75, iterations=152, cache=not EXPORT)
        hybrid = HybridRecommender([model_hybrid, user_cf, item_cbf, als],
                                   urm_train,
                                   merging_type=MergingTechniques.WEIGHTS,
                                   weights=[0.4767, 2.199, 2.604, 7.085],
                                   fallback_recommender=hybrid_fb)
        return hybrid

    def __init__(self, urm_train, users, weight_list):
        #assert len(weight_list) == max(users.values) + 1
        self.urm_train = urm_train
        self.users = users
        self.weight_list = weight_list
        self.rec_sys = self.initiliaze_recommender()

    def recommend(self, user_id, at=10, exclude_seen=True):
        user_group = users[user_id]
        self.rec_sys.weights = self.weight_list[user_group]
        return self.rec_sys.recommend(user_id, at, exclude_seen)


if __name__ == '__main__':
    set_seed(42)
    EXPORT = False
    urm, icm, ucm, target_users = build_all_matrices()
    if EXPORT:
        urm_train = urm.tocsr()
        urm_test = None
    else:
        urm_train, urm_test = train_test_split(urm, SplitType.PROBABILISTIC)

    urm, icm, ucm, target_users = build_all_matrices()


    _ , users = get_clusters_profile_length(urm_train, n_clusters=4)
    w_list = [
        [8.835, 9.287, 5.396, 9.454],   # cluster 1
        [5.938, 5.627, 2.84, 0.1773],   # cluster 2
        [0.4767, 2.199, 2.604, 7.085],  # cluster 3
        [3.458, 1.12, 0.7082, 6.993 ]   # cluster 4
    ]

    hybrid = GroupsHybridRecommender(urm_train=urm_train, users=users, weight_list=w_list )

    if EXPORT:
        export(target_users, hybrid)
    else:
        evaluate(hybrid, urm_test)
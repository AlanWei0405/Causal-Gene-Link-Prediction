import os
import pickle
from torch_geometric import transforms as T

from utils.meta2vec import get_features


def split_data(hetero_data_raw, runtime_type = 'heterogeneous'):

    train_path = './train.pkl'
    val_path = './val.pkl'
    test_path = './test.pkl'
    hetero_path = './hetero.pkl'

    if (((os.path.exists(train_path) and os.path.exists(val_path)) and
         (os.path.exists(test_path) and os.path.exists(hetero_path)))):
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)

        with open(val_path, 'rb') as f:
            val_data = pickle.load(f)

        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)

        with open(hetero_path, 'rb') as f:
            hetero_data = pickle.load(f)

    else:
        hetero_data = hetero_data_raw.coalesce()
        hetero_data = T.ToUndirected()(hetero_data)

        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.2,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=50.0,
            split_labels=True,
            add_negative_train_samples=True,
            edge_types=('gene', 'associate', 'disease'),
            rev_edge_types=('disease', 'rev_associate', 'gene')
        )

        train_data, val_data, test_data = transform(hetero_data)
        with open(train_path, 'wb') as f:
            pickle.dump(train_data, f)

        with open(val_path, 'wb') as f:
            pickle.dump(val_data, f)

        with open(test_path, 'wb') as f:
            pickle.dump(test_data, f)

        with open(hetero_path, 'wb') as f:
            pickle.dump(hetero_data, f)

    if runtime_type == 'single graph':
        # train_data = get_features(train_data)
        # val_data['gene'].x = train_data['gene'].x
        # test_data['gene'].x = train_data['gene'].x
        # val_data['disease'].x = train_data['disease'].x
        # test_data['disease'].x = train_data['disease'].x

        train_data= subset_to_homogeneous(train_data)
        val_data = subset_to_homogeneous(val_data)
        test_data = subset_to_homogeneous(test_data)

    return hetero_data, test_data, train_data, val_data


def subset_to_homogeneous(subset):
    subset_pos_edge_label = subset['gene', 'associate', 'disease'].pos_edge_label
    subset_pos_edge_label_index = subset['gene', 'associate', 'disease'].pos_edge_label_index
    subset_pos_edge_label_index[1] += subset['gene'].x.size(0)
    subset_neg_edge_label = subset['gene', 'associate', 'disease'].neg_edge_label
    subset_neg_edge_label_index = subset['gene', 'associate', 'disease'].neg_edge_label_index
    subset_neg_edge_label_index[1] += subset['gene'].x.size(0)

    del subset['gene', 'associate', 'disease'].pos_edge_label
    del subset['gene', 'associate', 'disease'].pos_edge_label_index
    del subset['gene', 'associate', 'disease'].neg_edge_label
    del subset['gene', 'associate', 'disease'].neg_edge_label_index

    subset = subset.to_homogeneous()

    subset.pos_edge_label = subset_pos_edge_label
    subset.pos_edge_label_index = subset_pos_edge_label_index
    subset.neg_edge_label = subset_neg_edge_label
    subset.neg_edge_label_index = subset_neg_edge_label_index

    return subset

import torch
# import torch.nn as nn
from torch_geometric.datasets import TUDataset, MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_networkx, get_laplacian

import matplotlib.pyplot as plt
import os
from os.path import isfile, join, abspath
import argparse
import numpy as np
from easydict import EasyDict as edict
import networkx as nx

# from Class_SciPySparseV2.utils import utils


def get_classes(dataset):
    class_labels = set()
    for data in dataset:
        class_labels.add(data.y.numpy()[0])
    return class_labels


def filter_num_samples(dataset, num_samples_per_class):
    filtered_data = []
    counts = {i: 0 for i in list(num_samples_per_class.keys())}  # Initialize counts for each class
    for data in dataset:
        target = data.y.numpy()[0]
        if counts[target] < num_samples_per_class[target]:
            filtered_data.append(data)
            counts[target] += 1
        if all(count == num_samples_per_class[target] for target, count in counts.items()):
            break  # Stop if we have reached the desired number of samples for each class
    return filtered_data


def data_loader(dataset_name: str = 'mnist',
                classes_list: str = [0, 1, 4],
                num_samp=10,
                seed=1,
                train: bool = True,
                show_info: bool = False,
                data_root_fold: str="./Data"):
    class MyFilter(object):
        def __init__(self, classes):
            self.classes = classes
        def __call__(self, data):
            return data.y in self.classes


    classes_str = '_'.join(map(str, classes_list))
    if dataset_name == 'mnist':
        dataset = MNISTSuperpixels(root='{:s}/Data/MNISTSuperpixels_dsseed={:03d}_cl={:s}'.format(data_root_fold, seed, classes_str),
                                   pre_filter=MyFilter(classes=classes_list),
                                   train=train)
    elif dataset_name == 'proteins':
        dataset = TUDataset(name='PROTEINS',  # transform=T.ToDense(max_nodes),
                            # pre_filter=MyFilter(),
                            root='./Data/TUDataset'
                            )
    elif dataset_name == 'mutag':
        dataset = TUDataset(root='data/TUDataset', name='MUTAG')

    else:
        raise SystemExit(f'No dataset named {dataset_name}')

    if show_info:
        print(f'Dataset: {dataset}:')
        print('============Dataset info ================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')

        data = dataset[0]  # Get the first graph object.
        print(data)
        print('================= Info for sample index 0 ============================================')
        # Gather some statistics about the first graph.
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Has self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}\n')

        # adj = to_dense_adj(data.edge_index).detach()[0] # data.edge_index is in COO format
        # g = to_networkx(data, to_undirected=True)
        # pos = data.pos.numpy()
        # pos[:, [1, 0]] = pos[:, [0, 1]]
        # pos_tuple = [(x, y) for (x, y) in pos]
        # print(pos.max(), pos.min())
        # nx.set_node_attributes(g, values=pos_tuple, name='coord')
        # fig, axes = plt.subplots(ncols=3, figsize=(18, 6))
        # nx.draw(g, pos=pos, node_color=data.x.numpy(), ax=axes[0])
        # axes[1].imshow(adj)
        # axes[1].set_title('Adj')
        # fig.suptitle(f'Label {data.y.numpy()}')
        # axes[2].hist(data.x.numpy())
        # axes[2].set_title('Node features')
        # plt.show()
        #

    torch.manual_seed(seed=seed)
    # Define the number of samples per class you want to keep
    num_samples_per_class = {class_label: num_samp for class_label in classes_list}
    dataset = dataset.shuffle()
    dataset = filter_num_samples(dataset=dataset, num_samples_per_class=num_samples_per_class)

    print('\nAfter filtering num_samples_per_class')
    print('\tDataset size:', len(dataset))
    print('\tClasses:', get_classes(dataset=dataset), '\n')

    return dataset

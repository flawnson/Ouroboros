from abc import ABC, abstractmethod

import dgl
import json
import torch
import numpy as np
import pandas as pd
import networkx as nx
import os.path as osp


class AbstractGraphDataset(ABC):
    """
    Abstract base class from which subclasses will inherit from.
    Currently the intended use is for datasets with multiple labelsets.
    Modifications can be made to compensate for multiple graph edgelists, node features, etc.
    """
    @abstractmethod
    def __init__(self, config: dict):
        super(AbstractGraphDataset, self).__init__()
        self.config = config
        self.data_config = self.config["data_config"]
        self.dataset = self.preprocessing()

    def get_edges(self) -> nx.DiGraph:
        edgelist_path = osp.join(osp.dirname(osp.dirname(__file__)),
                                 osp.join(*self.data_config["directory"]),
                                 self.data_config["edgelist_file"])

        # For PyG
        edgeframe = pd.read_csv(edgelist_path, header=0, index_col=False)

        # DGL will turn graph into directed graph regardless of networkx object type
        edges = nx.readwrite.edgelist.read_edgelist(edgelist_path, delimiter=",", create_using=nx.DiGraph)

        return edges

    def get_features(self, features_filename) -> dict:
        features_dict = json.load(open(osp.join(osp.dirname(osp.dirname(__file__)),
                                                *self.data_config["directory"],
                                                features_filename)))

        features_dict = {name: embed for name, embed
                         in zip(features_dict["gene"].values(),
                                features_dict["embeddings"].values())}

        if self.data_config["dummy_features"]:
            features_dict = {key: np.ones(self.data_config["dummy_features"], dtype=np.double) for key, value
                             in features_dict.items()}

        return features_dict

    def get_targets(self) -> dict:
        node_labels = pd.read_csv(osp.join(osp.dirname(osp.dirname(__file__)),
                                           *self.data_config["directory"],
                                           self.data_config["label_file"]), header=0)

        target_data = [node_labels[name].tolist() for name in self.data_config["label_names"]]

        return self.get_labels(target_data)

    @abstractmethod
    def get_labels(self, target_data: list):
        # Must be implemented by subclasses. File reading logic is held in get_targets()
        return {}

    def intersection(self) -> nx.DiGraph:
        nx_graph: nx.graph = self.get_edges()
        target_data: dict = self.get_targets()

        # Needed to compensate for differences between target set and edgelist (assign nodes without labels to unknown)
        target_data = {name: 0 if name not in target_data else target_data[name] for name in nx_graph.nodes()}

        nx.set_node_attributes(nx_graph, target_data, "y")

        if self.data_config["node_features_file"]:
            nx_graph.node_data = ["x"]
            nx.set_node_attributes(nx_graph, self.get_features(self.data_config["node_features_file"]), *nx_graph.node_data)
        else:
            nx_graph.node_data = None

        if self.data_config["edge_features_file"]:
            nx_graph.edge_data = ["z"]
            nx.set_edge_attributes(nx_graph, self.get_features(self.data_config["edge_features_file"]), *nx_graph.edge_data)
        else:
            nx_graph.edge_data = None

        # Filter for nodes with embeddings (inplace operation does not need variable assignment
        [nx_graph.remove_node(n) for (n, d) in nx_graph.copy().nodes(data=True) if "x" not in d]

        return nx_graph

    def preprocessing(self) -> dgl.graph:
        # TODO: Check if node list is getting rearranged during conversion to dgl graph object
        nx_graph = self.intersection()
        if self.data_config["visualize"]:
            VisualizeData(self.config, nx_graph).draw()
        # dgl_graph = dgl.DGLGraph()
        dgl_graph = dgl.from_networkx(nx_graph, node_attrs=nx_graph.node_data + ["y"], edge_attrs=nx_graph.edge_data)

        # dgl_graph.y = nx_graph.nodes("y")  # PyG's preferred method of adding attributes to object class

        # Creating known node mask for semi-supervised task
        dgl_graph.known_mask = dgl_graph.ndata["y"].numpy() != 0 if self.data_config["semi-supervised"] else torch.ones(
            len(dgl_graph.y))

        return dgl_graph


class PrimaryLabelset(AbstractGraphDataset, ABC):
    def __init__(self, config: dict):
        super(PrimaryLabelset, self).__init__(config=config)

    @staticmethod
    def extract_labels(target_data) -> dict:
        # Labels starts from 1, since 0 is reserved for unknown class in the case of semi-supervised learning
        # Can manually create dictionary that maps from data to integer
        # Note that PyG automatically turns ints into onehot
        return dict(zip(np.unique(target_data[1]), list(range(1, len(np.unique(target_data[1])) + 1))))

    def get_labels(self, target_data) -> dict:
        # Abstract method defined in GenericDataset
        return {name: self.extract_labels(target_data)[label] for name, label in zip(target_data[0], target_data[1])}
        # return [self.extract_labels(target_data)[label] for name, label in zip(target_data[0], target_data[1])]
import torch
from torch_geometric.nn import GCNConv
from torch.nn import Dropout, ReLU, Linear, Parameter
from torch_geometric.nn import global_max_pool
import numpy as np
from sklearn.cluster import AgglomerativeClustering


class UTango(torch.nn.Module):
    def __init__(self, h_size, drop_out_rate, max_context, gcn_layers):
        super(UTango, self).__init__()
        self.gcn_layers = gcn_layers
        self.h_size = h_size
        self.drop_out_rate = drop_out_rate
        self.max_context = max_context
        self.GCN = GCNConv(self.h_size, self.h_size)
        self.dropout = Dropout(self.drop_out_rate)
        self.relu = ReLU(inplace=True)
        self.resize = Linear(self.h_size*self.max_context, self.h_size)
        self.threshold = Parameter(torch.zeros(1))

    def forward(self, input_data):
        pre_clu = []
        for data in input_data:
            node_features = data.x
            node_label = data.y
            edge_info = data.edge_index

            for i in range(self.gcn_layers - 1):
                feature_vec = self.GCN(node_features, edge_info)
                feature_vec = self.relu(feature_vec)
                feature_vec = self.dropout(feature_vec)
            feature_vec = self.GCN(feature_vec, edge_info)

            output = []
            for i in range(len(node_label)):
                if node_label[i] != 0:
                    context_vec = []
                    for j in range(self.max_context):
                        if j < len(node_label[i]):
                            context_vec.append(feature_vec[node_label[i][j]])
                        else:
                            context_vec.append(torch.zeros(self.h_size))
                    context_vec = torch.concat(context_vec)
                    context_vec = self.resize(context_vec)
                    rep_vec = torch.reshape(feature_vec[i], (-1,)) * context_vec
                    output.append(rep_vec.tolist())
            threshold = torch.sigmoid(self.threshold)
            clustering = AgglomerativeClustering(distance_threshold=threshold.tolist(), n_clusters=None)
            if len(output) == 1:
                clu_labels = np.zeros(1)
            else:
                clustering.fit(output)
                clu_labels = clustering.labels_
            pre_clu.append(clu_labels)
        return pre_clu


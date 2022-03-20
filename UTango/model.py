import torch
from torch_geometric.nn import GCNConv
from torch.nn import Dropout, ReLU, Linear, Softmax
import networkx


class Utango(torch.nn.Module):
    def __init__(self, h_size, drop_out_rate, max_method, gcn_layers):
        super(Utango, self).__init__()
        self.gcn_layers = gcn_layers
        self.h_size = h_size
       # self.max_context = max_context
        self.drop_out_rate = drop_out_rate
        self.max_method = max_method
        self.GCN_1 = GCNConv(self.h_size, self.h_size)
        self.GCN_2 = GCNConv(self.h_size, self.h_size)
        self.GCN_3 = GCNConv(self.h_size, self.h_size)
        self.GCN_4 = GCNConv(self.h_size, self.h_size)
        self.GCN_5 = GCNConv(self.h_size, self.h_size)
        self.GCN_6 = GCNConv(self.h_size, self.h_size)
        self.GCN_7 = GCNConv(self.h_size, self.h_size)
        self.dropout = Dropout(self.drop_out_rate)
        self.relu = ReLU(inplace=True)
        self.resize_1 = Linear(self.h_size, 3)
        self.resize_2 = Linear(self.h_size, 3)
        self.resize_3 = Linear(self.h_size, 3)
        self.resize_4 = Linear(self.h_size, 3)
        self.resize_5 = Linear(self.h_size, 3)
        self.resize_6 = Linear(self.h_size, 2)
        self.resize_7 = Linear(self.h_size, 2)
        self.resize_m = Linear(self.max_method*self.h_size, self.h_size)
        self.softmax = Softmax(dim=1)

    def forward(self, data):
        node_features = data.x
        vul_ass_types = data.y
        edge_info = data.edge_index

        for i in range(self.gcn_layers - 1):
            feature_vec1 = self.GCN_1(node_features, edge_info)
            feature_vec1 = self.relu(feature_vec1)
            feature_vec1 = self.dropout(feature_vec1)
        feature_vec1 = self.GCN_1(feature_vec1, edge_info)

        for i in range(self.gcn_layers - 1):
            feature_vec2 = self.GCN_2(node_features, edge_info)
            feature_vec2 = self.relu(feature_vec2)
            feature_vec2 = self.dropout(feature_vec2)
        feature_vec2 = self.GCN_2(feature_vec2, edge_info)

        for i in range(self.gcn_layers - 1):
            feature_vec3 = self.GCN_3(node_features, edge_info)
            feature_vec3 = self.relu(feature_vec3)
            feature_vec3 = self.dropout(feature_vec3)
        feature_vec3 = self.GCN_3(feature_vec3, edge_info)

        for i in range(self.gcn_layers - 1):
            feature_vec4 = self.GCN_4(node_features, edge_info)
            feature_vec4 = self.relu(feature_vec4)
            feature_vec4 = self.dropout(feature_vec4)
        feature_vec4 = self.GCN_4(feature_vec4, edge_info)

        for i in range(self.gcn_layers - 1):
            feature_vec5 = self.GCN_5(node_features, edge_info)
            feature_vec5 = self.relu(feature_vec5)
            feature_vec5 = self.dropout(feature_vec5)
        feature_vec5 = self.GCN_5(feature_vec5, edge_info)

        for i in range(self.gcn_layers - 1):
            feature_vec6 = self.GCN_6(node_features, edge_info)
            feature_vec6 = self.relu(feature_vec6)
            feature_vec6 = self.dropout(feature_vec6)
        feature_vec6 = self.GCN_6(feature_vec6, edge_info)

        for i in range(self.gcn_layers - 1):
            feature_vec7 = self.GCN_7(node_features, edge_info)
            feature_vec7 = self.relu(feature_vec7)
            feature_vec7 = self.dropout(feature_vec7)
        feature_vec7 = self.GCN_7(feature_vec7, edge_info)

        vec_1 = self.resize_1(feature_vec1)
        vec_1 = self.softmax(vec_1)

        vec_2 = self.resize_2(feature_vec2)
        vec_2 = self.softmax(vec_2)

        vec_3 = self.resize_3(feature_vec3)
        vec_3 = self.softmax(vec_3)

        vec_4 = self.resize_4(feature_vec4)
        vec_4 = self.softmax(vec_4)

        vec_5 = self.resize_5(feature_vec5)
        vec_5 = self.softmax(vec_5)

        vec_6 = self.resize_6(feature_vec6)
        vec_6 = self.softmax(vec_6)

        vec_7 = self.resize_7(feature_vec7)
        vec_7 = self.softmax(vec_7)

        return [vec_1, vec_2, vec_3, vec_4, vec_5, vec_6, vec_7]

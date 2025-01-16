
import torch
from torch import nn
import torch_scatter




def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    # trans 边的坐标差 边X3, row 边条数 边, num_segments = coord.size(0) 节点数 int
    result_shape = (num_segments, data.size(1)) ##(num_nodes, 3)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1)) #(num_edges, 3)

    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0) #初始化两个全为0的(num_nodes, 3)矩阵

    result.scatter_add_(0, segment_ids, data) #
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    # 这两行就是根据边的索引把边的坐标差累加到对应节点上面去
    # 第二行是创建一个(num_edges, 3)的全1矩阵，统计每一个节点被累加了多少次，第一个0是指按第0维操作

    return result / count.clamp(min=1) # 计算每个分组的均值，并防止除0


class E_GCL(nn.Module):

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()


        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hidden_nf+17, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:               # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1) # 1024，1024，1，16 > 2065
        
        out = out.to(torch.float32)
        out = self.edge_mlp(out) #2065 >1024
        
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        # x：节点特征，edge_attr：边特征
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0)) #nodeX1024

        #sys.exit()
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        
        agg = agg.to(torch.float32)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) #先把边的特征映射成一个值，再把坐标差X这个值

        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        #这里相当于根据边的信息算节点的聚合
        coord = coord + agg

        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index[0], edge_index[1]
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1) # 计算径向距离（欧几里得距离的平方）(Batch_size, 1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        # h: node的embedding，edge_index：边的index，coord：原子坐标，edge_attr：边的径向基函数特征
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr) #相当于把边的两个节点的特征，边的距离和径向基函数特征融合起来了
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr




class EGNNModel(nn.Module):
    def __init__(self, config):
        super(EGNNModel, self).__init__()

        global in_edge_nf, max_len, n_layers, n_head, d_model, d_ff, vocab_size, device, act_fn, residual, attention, normalize, tanh, hidden_nf, drop_value

        in_edge_nf = config.in_edge_nf
        n_layers = config.num_layer
        d_model = config.dim_embedding
        device = config.device
        drop_value = config.dropout
        act_fn = nn.SiLU()
        residual = True
        attention = False
        normalize = False
        tanh = False
        hidden_nf = d_model


        self.norm = nn.LayerNorm(d_model)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i,
                            E_GCL(hidden_nf, hidden_nf, hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn,residual=residual, attention=attention,normalize=normalize, tanh=tanh))

        
        self.block1 = nn.Sequential(
            nn.Linear(hidden_nf,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
# #             nn.Dropout(0.3),
            nn.Linear(256, 128),
            )

        self.block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 2),
        )
        
        
        self.W_v = nn.Linear(1069, hidden_nf, bias=True)


    def forward(self, batch):
        x, h = batch.x, batch.plm
        h = self.W_v(h)

        for i in range(0, n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, batch.edge_index, x, edge_attr=batch.edge_s) #每一次传进去的边特征不变？

        out = torch_scatter.scatter_max(h, batch.batch, dim=0)[0].float()
        representations = self.block1(out)

        return representations

    
    def get_logits(self, batch):
        with torch.no_grad():
            output = self.forward(batch)
        logits = self.block2(output)

        return logits




import matplotlib.pyplot as plt
import warnings
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy
from sklearn.metrics import roc_auc_score


# rating_col = ['user_id', 'movie_id', 'age', 'gender', 'occupation', 'genre', 'rating', 'advertiserIdx']


# 计算AUC
def get_auc(y_pred, y_true):
    y_pred = y_pred.data.detach().cpu().numpy()
    y_true = y_true.data.detach().cpu().numpy()
    return roc_auc_score(y_true, y_pred)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Embedding') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)
    if classname.find("DomainLayer") != -1 or classname.find("CrossPnn") != -1:
        for key, val in m.state_dict().items():
            if key.find("bias") != -1:
                torch.nn.init.constant_(val, 0)
            else:
                torch.nn.init.xavier_normal_(val)

    if classname.find("myNet") != -1:
        for key, val in m.state_dict().items():
            if key.find("bias") != -1:
                torch.nn.init.constant_(val, 0)
            else:
                torch.nn.init.xavier_normal_(val)


def average_all_layers(nets):
    state_dict = [x.state_dict() for x in nets]
    keys = list(state_dict[0].keys())
    for k in keys:
        weights = 0
        for i in range(len(nets)):
            weights = weights + state_dict[i][k]
        for net in nets:
            net.state_dict()[k].copy_(weights / len(nets))


def average_other_feas(nets):
    print(" -------------------------------- average other feas:")
    rating_col = ['age', 'gender', 'occupation', 'genre']
    state_dict = [x.state_dict() for x in nets]
    keys = list(state_dict[0].keys())
    for k in keys:
        for clo in rating_col:
            if k.__contains__(clo):
                weights = 0
                for i in range(len(nets)):
                    weights = weights + state_dict[i][k]
                for net in nets:
                    net.state_dict()[k].copy_(weights / len(nets))
    return nets


from sklearn.decomposition import PCA


def agg_emb_pca(embs):
    emb = np.array(embs).T
    pca = PCA(n_components=1)
    pca.fit(emb)
    emb = torch.tensor(pca.transform(emb).T).squeeze(0)
    return emb


# Python实现PCA
def agg_emb_my_pca(embs):
    # k is the components you want
    X = np.array(embs).T
    # mean of each feature
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    # normalization
    norm_X = X - mean
    # scatter matrix
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    # eig_pairs.sort(reverse=True)
    eig_pairs.sort(reverse=False)  # lowest to highest
    # select the top k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:1]])
    # get new data
    data = np.dot(norm_X, np.transpose(feature))
    emb = torch.tensor(data.T).squeeze(0)
    return emb


# https://www.zhihu.com/question/419103746/answer/1450319683
def agg_emb_my_pca2(embs):
    x = np.array(embs).T
    x -= x.mean(axis=0)
    C = x.T.dot(x)  # 计算自协方差矩阵
    lam, v = np.linalg.eig(C)  # 特征值和特征向量
    # 默认为从小到大的对应的索引
    # 从大到小的对应的索引，等价于np.argsort(a)[::-1]
    new_index = np.argsort(lam)[::-1]
    A = -v[:, new_index]
    W = x.dot(A)  # 计算变换后的矩阵
    R = lam[new_index] / lam.sum()  # 计算所有特征对应贡献率
    emb = torch.tensor(W.T[0]).squeeze(0)
    agg_emb = []
    for r, w in zip(R, W.T):
        agg_emb.append(r * w)
    emb = torch.mean(torch.tensor(agg_emb), dim=0, keepdim=True).squeeze(0)
    return emb


def agg_emb_avg(embs):
    avg_emb = torch.mean(torch.tensor(embs), dim=0, keepdim=True).squeeze(0)
    return avg_emb


def get_agg_emb(embs, strategy='my_pca'):
    if len(embs) > 1:
        if strategy == 'avg':
            return agg_emb_avg(embs)
        elif strategy == 'my_pca':
            return agg_emb_my_pca2(embs)
        elif strategy == 'pca':
            return agg_emb_pca(embs)
        else:
            warnings.warn('agg stragety wrong', UserWarning)


def set_agg_emb(nets, uid, sid, agg_emb, emb_name):
    for id in sid:
        nets[id].state_dict()[emb_name][uid].copy_(agg_emb)


def agg_user_emb(nets, dData, use_cuda, strategy):
    print(" -------------------------------- u agg strategy:", strategy)
    u_features = ['user_id', 'age', 'gender', 'occupation', 'serverIdx']
    fea = u_features[0]
    emb_name = "net.embed_layers.embed_" + fea + ".weight"
    # if modle == 'mlp':
    #     emb_name = "net.embed_" + fea + ".weight"
    user_id, server_id = dData['user_id'], dData[u_features[-1]]
    for uid, sid in zip(user_id, server_id):
        uEmbs = []
        # 获取对应Server net的uemb
        if len(sid) > 1:
            for id in sid:
                emb = nets[id].state_dict()[emb_name][uid]
                if use_cuda:
                    emb = emb.cpu()
                uEmbs.append(emb.numpy())
            # 对emb聚合
            agg_emb = get_agg_emb(uEmbs, strategy)
            # 更新 Server net 的uemb
            set_agg_emb(nets, uid, sid, agg_emb, emb_name)


def agg_item_emb(nets, aData, use_cuda, strategy):
    print(" -------------------------------- v agg strategy:", strategy)
    v_features = ['movie_id', 'genre', 'serverIdx']
    fea = v_features[0]
    emb_name = "net.embed_layers.embed_" + fea + ".weight"

    #for aData in aDatas:
    movie_id, server_id = aData['movie_id'], aData[v_features[-1]]
    for vid, sid in zip(movie_id, server_id):
        if len(sid) > 1:
            vEmbs = []
            for id in sid:
                emb = nets[id].state_dict()[emb_name][vid]
                if use_cuda:
                    emb = emb.cpu()
                vEmbs.append(emb.numpy())
            agg_emb = get_agg_emb(vEmbs, strategy)
            set_agg_emb(nets, vid, sid, agg_emb, emb_name)


"=================================================================================================================================================readdata"

import itertools
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder


def sda_dataloader(data, batch_size, use_cuda, rating_col):
    # 4以上转换为positive，其他转换为negative
    # rating_col = ['user_id', 'movie_id', 'age', 'gender', 'occupation', 'genre', 'rating']
    data['rating'] = np.where(data['rating'] > 3, 1, 0)
    data_x = data[rating_col]
    data_y = data['rating']
    x_tensor = torch.tensor(data_x.values, dtype=torch.float32)
    y_tensor = torch.tensor(data_y.values, dtype=torch.float32)
    if use_cuda:
        x_tensor = x_tensor.cuda()
        y_tensor = y_tensor.cuda()
    # 创建数据集
    dataset = TensorDataset(x_tensor, y_tensor)
    train_data = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    return train_data


def readdata_SDA_ML1M(root_path, fileS, fileA, batch_size=512, use_cuda=False, test=False):
    # d: user_id,gender,age,occupation,serverIdx
    # 全部数据
    rating_col = ['user_id', 'movie_id', 'age', 'gender', 'occupation', 'genre', 'rating', 'advertiserIdx']
    allrating = pd.read_csv(root_path + 'allrating.csv')
    fea_dims = {}
    for fea in rating_col[:-2]:
        fea_dims[fea] = allrating[fea].max() + 1
    del allrating
    # s: user_id,movie_id,age,gender,occupation,genre,rating,advertiserIdx
    valid_data = pd.read_csv(root_path + 'valid.csv')  # , converters={'genre': literal_eval})
    test_data = pd.read_csv(root_path + 'test.csv')  # , converters={'genre': literal_eval})
    test_data = test_data[:1000] if test else test_data
    valid_data = valid_data[:1000] if test else valid_data

    sDataTrain, sDataValid, aDatas, dData = [], [], [], []
    for file in fileS:
        file = root_path + file + '.csv'
        data = pd.read_csv(file)
        data = data[:6000] if test else data
        valid = data.sample(frac=0.1, random_state=0)
        train = data.drop(valid.index)

        sDataTrain.append(train)
        sDataValid.append(valid)

    import ast
    # a: movie_id,genre,serverIdx
    aColumn = ['movie_id', 'genre', 'serverIdx']
    # for file in fileA:
    #     file = root_path + file + '.csv'
    #     data = pd.read_csv(file, usecols=aColumn, converters={'serverIdx': ast.literal_eval})
    #     aDatas.append(data)
    aData = pd.read_csv(root_path + 'a.csv', usecols=aColumn, converters={'serverIdx': ast.literal_eval})

    # user_id,gender,age,occupation,serverIdx
    dColumn = ['user_id', 'gender', 'age', 'occupation', 'serverIdx']
    dData = pd.read_csv(root_path + 'd.csv', usecols=dColumn, converters={'serverIdx': ast.literal_eval})
    # create dataloader
    for i in range(len(sDataTrain)):
        print("len s_", i, ":", len(sDataTrain[i]))
        sDataTrain[i] = sda_dataloader(sDataTrain[i], batch_size, use_cuda, rating_col[:-2])
        sDataValid[i] = sda_dataloader(sDataValid[i], batch_size, use_cuda, rating_col[:-2])

    valid_data = sda_dataloader(valid_data, batch_size, use_cuda, rating_col[:-2])
    test_data = sda_dataloader(test_data, batch_size, use_cuda, rating_col[:-2])
    return sDataTrain, sDataValid, aData, dData, valid_data, test_data, fea_dims


import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


import torch.nn as nn
import os

os.environ['CUDA_ENABLE_DEVICES'] = '0'


# 定义一个全连接层的神经网络
class DNN(nn.Module):
    def __init__(self, hiddenUnits, dropout=0.3):
        super(DNN, self).__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hiddenUnits[:-1], hiddenUnits[1:]))])
        self.dropout = nn.Dropout(p=dropout)

    # 前向传播， 遍历dnn_network， 加激活函数
    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        x = self.dropout(x)
        return x


# feaInfo={name:feaName, key:maxNum}
class MLP(nn.Module):
    def __init__(self, embInfo, hiddenUnits, lossname, emb_dim):
        super(MLP, self).__init__()
        self.lossname = lossname
        self.embInfo = embInfo
        self.embDim = emb_dim
        # self.embed_user_id = nn.Embedding(num_embeddings=embInfo['user_id'], embedding_dim=emb_dim)
        # self.embed_movie_id = nn.Embedding(num_embeddings=embInfo['movie_id'], embedding_dim=emb_dim)
        # embedding层，类别特征embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=self.embDim)
            for key, val in self.embInfo.items()
        })

        """ fully connected layer """
        # self.MLP_Layers = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(Layers[:-1], Layers[1:]))])
        self.dnn_network = DNN(hiddenUnits)
        self.dense_final = nn.Linear(hiddenUnits[-1], 1)
        self.get_auc = get_auc

        self.lossfunc = torch.nn.BCELoss()
        self.actifunc = torch.nn.Sigmoid()
        self.metricfunc = {"auc": get_auc}

    def forward(self, inputs):
        """ 嵌入 """
        inputs = inputs.long()
        inputs_embeds = [self.embed_layers['embed_' + key](inputs[:, i])
                         for key, i in zip(self.embInfo.keys(), range(inputs.shape[1]))]

        # embed_user_id = self.embed_user_id(inputs[:, 0])
        # embed_movie_id = self.embed_movie_id(inputs[:, 1])
        """ 拼接 """
        embedding_cat = torch.cat(inputs_embeds, dim=-1)
        # embedding_cat = torch.cat((embed_user_id, embed_movie_id), dim=1)
        # """ 点乘 """
        # embedding_vec = torch.mul(embed_user_id, embed_movie_id)
        """ 全连接 """
        # for mlp in self.MLP_Layers:
        #     embedding_vec = mlp(embedding_vec)
        #     embedding_vec = F.relu(embedding_vec)
        dnn_x = self.dnn_network(embedding_cat)
        # outputs = torch.sigmoid(self.dense_final(dnn_x))
        outputs = self.dense_final(dnn_x)
        outputs = self.actifunc(outputs)
        outputs = outputs.view(-1)
        return outputs


class MetaRecommender(nn.Module):  # in fact, it's not a hypernetwork
    def __init__(self, user_num, item_num, item_emb_size=32, item_mem_num=8, user_emb_size=32, metaMemory=64,
                 hidden_size=64):  # note that we have many users and each user has many layers
        super(MetaRecommender, self).__init__()
        self.item_num = item_num
        self.item_emb_size = item_emb_size
        self.item_mem_num = item_mem_num
        # For each user
        self.embed_user_id = nn.Embedding(user_num, user_emb_size)
        self.memory = torch.nn.Parameter(nn.init.xavier_normal_(torch.Tensor(user_emb_size, metaMemory)),
                                         requires_grad=True)
        # For each layer
        self.hidden_layer_1, self.weight_layer_1, self.bias_layer_1 = self.define_one_layer(metaMemory, hidden_size,
                                                                                            item_emb_size,
                                                                                            int(item_emb_size / 4))
        self.hidden_layer_2, self.weight_layer_2, self.bias_layer_2 = self.define_one_layer(metaMemory, hidden_size,
                                                                                            int(item_emb_size / 4), 1)
        self.hidden_layer_3, self.emb_layer_1, self.emb_layer_2 = self.define_item_embedding(item_num, item_emb_size,
                                                                                             item_mem_num, metaMemory,
                                                                                             hidden_size)

    def define_one_layer(self, metaMemory, hidden_size, int_size, out_size):  # define one layer in MetaMF
        hidden_layer = nn.Linear(metaMemory, hidden_size)
        weight_layer = nn.Linear(hidden_size, int_size * out_size)
        bias_layer = nn.Linear(hidden_size, out_size)
        return hidden_layer, weight_layer, bias_layer

    def define_item_embedding(self, item_num, item_emb_size, item_mem_num, metaMemory, hidden_size):
        hidden_layer = nn.Linear(metaMemory, hidden_size)
        emb_layer_1 = nn.Linear(hidden_size, item_num * item_mem_num)
        emb_layer_2 = nn.Linear(hidden_size, item_mem_num * item_emb_size)
        return hidden_layer, emb_layer_1, emb_layer_2

    def forward(self, user_id):
        # collaborative memory module
        user_emb = self.embed_user_id(user_id)  # input_user=[batch_size, user_emb_size]
        cf_vec = torch.matmul(user_emb,
                              self.memory)  # [user_num, u_emb_size]*[u_emb_size, metaMemory]=[batch_size, metaMemory]
        output_weight = []
        output_bias = []
        weight, bias = self.get_one_layer(self.hidden_layer_1, self.weight_layer_1, self.bias_layer_1, cf_vec,
                                          self.item_emb_size, int(self.item_emb_size / 4))
        output_weight.append(weight)
        output_bias.append(bias)
        weight, bias = self.get_one_layer(self.hidden_layer_2, self.weight_layer_2, self.bias_layer_2, cf_vec,
                                          int(self.item_emb_size / 4), 1)
        output_weight.append(weight)
        output_bias.append(bias)
        item_embedding = self.get_item_embedding(self.hidden_layer_3, self.emb_layer_1, self.emb_layer_2, cf_vec,
                                                 self.item_num, self.item_mem_num, self.item_emb_size)
        # meta recommender module
        return output_weight, output_bias, item_embedding, cf_vec  # ([len(layer_list)+1, batch_size, *, *], [len(layer_list)+1, batch_size, 1, *], [batch_size, item_num, item_emb_size], [batch_size, metaMemory])

    def get_one_layer(self, hidden_layer, weight_layer, bias_layer, cf_vec, int_size,
                      out_size):  # get one layer in MetaMF
        hid = hidden_layer(cf_vec)  # hid=[batch_size, hidden_size]
        hid = F.relu(hid)
        weight = weight_layer(hid)  # weight=[batch_size, self.layer_list[i-1]*self.layer_list[i]]
        bias = bias_layer(hid)  # bias=[batch_size, self.layer_list[i]]
        weight = weight.view(-1, int_size, out_size)
        bias = bias.view(-1, 1, out_size)
        return weight, bias

    def get_item_embedding(self, hidden_layer, emb_layer_1, emb_layer_2, cf_vec, item_num, item_mem_num, item_emb_size):
        hid = hidden_layer(cf_vec)  # hid=[batch_size, hidden_size]
        hid = F.relu(hid)
        emb_left = emb_layer_1(hid)  # emb_left=[batch_size, item_num*item_mem_num]
        emb_right = emb_layer_2(hid)  # emb_right=[batch_size, item_mem_num*item_emb_size]
        emb_left = emb_left.view(-1, item_num, item_mem_num)  # emb_left=[batch_size, item_num, item_mem_num]
        emb_right = emb_right.view(-1, item_mem_num,
                                   item_emb_size)  # emb_right=[batch_size, item_mem_num, item_emb_size]
        item_embedding = torch.matmul(emb_left, emb_right)  # item_embedding=[batch_size, item_num, item_emb_size]
        return item_embedding


class MetaMF(nn.Module):
    def __init__(self, embInfo, lossname, item_emb_size=10, item_mem_num=8, user_emb_size=10, metaMemory=64,
                 hidden_size=64):
        super(MetaMF, self).__init__()
        self.item_num = embInfo['movie_id']
        self.metarecommender = MetaRecommender(embInfo['user_id'], embInfo['movie_id'], item_emb_size,
                                               item_mem_num, user_emb_size, metaMemory, hidden_size)
        self.lossname = lossname
        self.lossfunc = torch.nn.BCELoss()
        self.actifunc = torch.nn.Sigmoid()
        self.metricfunc = {"auc": get_auc}

    def forward(self, inputs):
        # prediction module
        inputs = inputs.long()
        TorchKerasModelCopy_weight, TorchKerasModelCopy_bias, item_embedding, _ = self.metarecommender(inputs[:, 0])
        item_id = inputs[:, 1]
        item_id = item_id.view(-1, 1)  # item_id=[batch_size, 1]
        item_one_hot = torch.zeros(len(item_id), self.item_num,
                                   device=item_id.device)  # we generate it dynamically, and default device is cpu
        item_one_hot.scatter_(1, item_id, 1)  # item_one_hot=[batch_size, item_num]
        item_one_hot = torch.unsqueeze(item_one_hot, 1)  # item_one_hot=[batch_size, 1, item_num]
        item_emb = torch.matmul(item_one_hot, item_embedding)  # out=[batch_size, 1, item_emb_size]
        out = torch.matmul(item_emb, TorchKerasModelCopy_weight[0])  # out=[batch_size, 1, item_emb_size/4]
        out = out + TorchKerasModelCopy_bias[0]  # out=[batch_size, 1, item_emb_size/4]
        out = F.relu(out)  # out=[batch_size, 1, item_emb_size/4]
        out = torch.matmul(out, TorchKerasModelCopy_weight[1])  # out=[batch_size, 1, 1]
        out = out + TorchKerasModelCopy_bias[1]  # out=[batch_size, 1, 1]
        out = self.actifunc(out)
        out = torch.squeeze(out)  # out=[batch_size]
        out = out.view(-1)
        # prediction module
        return out


class CrossDcn(nn.Module):
    def __init__(self, layer_num, input_dim):
        super(CrossDcn, self).__init__()
        self.layer_num = layer_num
        # 定义网络层的参数
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.rand(input_dim, 1))
            for i in range(self.layer_num)
        ])
        self.cross_bias = nn.ParameterList([
            nn.Parameter(torch.rand(input_dim, 1))
            for i in range(self.layer_num)
        ])

    def forward(self, x):
        # x是(None, dim)的形状， 先扩展一个维度到(None, dim, 1)
        x = torch.unsqueeze(x, dim=2)  # [128,20]->[128,2*10, 1]
        xC = x.clone()  # [128,20, 1]
        xT = x.clone().permute((0, 2, 1))  # [128,1,20]
        for i in range(self.layer_num):
            x = torch.matmul(torch.bmm(xC, xT), self.cross_weights[i]) + self.cross_bias[i] + x  # [128, 20, 1)
            xT = x.clone().permute((0, 2, 1))  # (None, 1, dim)

        x = torch.squeeze(x)  # (None, dim)
        return x


class DCN(nn.Module):
    def __init__(self, embInfo, hiddenUnits, layer_num, use_cuda, lossname, embDim):
        super(DCN, self).__init__()
        # self.dense_embInfo, self.embInfo = embInfo
        self.embInfo = embInfo
        # embedding层，类别特征embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=embDim)
            for key, val in embInfo.items()
        })

        # hiddenUnits.insert(0, len(embInfo) * embDim)

        self.cross_network = CrossDcn(layer_num, hiddenUnits[0])  # layer_num是交叉网络的层数， hiddenUnits[0]表示输入的整体维度大小
        self.dnn_network = DNN(hiddenUnits)
        self.dense_final = nn.Linear(hiddenUnits[-1] + hiddenUnits[0], 1)
        self.lossname = lossname

        self.lossfunc = torch.nn.BCELoss()
        self.actifunc = torch.nn.Sigmoid()
        self.metricfunc = {"auc": get_auc}

    def forward(self, x):
        inputs = x.long()
        inputs_embeds = [self.embed_layers['embed_' + key](inputs[:, i])
                         for key, i in zip(self.embInfo.keys(), range(inputs.shape[1]))]
        embeds = torch.cat(inputs_embeds, dim=-1)
        # cross Network
        cross_out = self.cross_network(embeds)
        # Deep Network
        deep_out = self.dnn_network(embeds)
        #  Concatenate
        total_x = torch.cat([cross_out, deep_out], 1)
        # out
        outputs = self.dense_final(total_x)
        outputs = self.actifunc(outputs)
        outputs = outputs.view(-1)
        return outputs


class CrossPnn(nn.Module):

    def __init__(self, embDim, sparse_num, hiddenUnits, use_cuda):
        super(CrossPnn, self).__init__()
        self.w_z = nn.Parameter(torch.rand([sparse_num, embDim, hiddenUnits[0]]))
        # p部分
        self.w_p = nn.Parameter(torch.rand([sparse_num, sparse_num, hiddenUnits[0]]))  # [26,26,256]
        self.l_b = torch.rand([hiddenUnits[0], ], requires_grad=True)
        if use_cuda:
            self.l_b = self.l_b.cuda()

    def forward(self, z, inputs_embeds):
        # l_z = torch.mm([128,70], [70,64])=[128,64]
        l_z = torch.mm(z.reshape(z.shape[0], -1),
                       self.w_z.permute((2, 0, 1)).reshape(self.w_z.shape[2], -1).T)  # (None, hiddenUnits[0])
        # matmul可处理维度不同的矩阵, : [2,5,3]*[1,3,4]->[2,5,4]
        # lp = matmul([128,7,10], [128, 10, 7])=[128, 7, 7]
        p = torch.matmul(inputs_embeds, inputs_embeds.permute((0, 2, 1)))  # [None, sparse_num, sparse_num]
        # mm([128,49],[49,64])=[128,64]
        l_p = torch.mm(p.reshape(p.shape[0], -1),
                       self.w_p.permute((2, 0, 1)).reshape(self.w_p.shape[2], -1).T)  # [None, hiddenUnits[0]]
        # output = [128,64]+[128,64]+[64]=[128,64]
        output = l_z + l_p + self.l_b
        return output


# PNN网络
# 逻辑是底层输入（类别型特征) -> embedding层 -> crosspnn 层 -> DNN -> 输出
class PNN(nn.Module):
    # hiddenUnits = [256, 128, 64]
    def __init__(self, embInfo, hiddenUnits, use_cuda, lossname, embDim, mode_type='in', dnn_dropout=0.3):
        super(PNN, self).__init__()
        self.features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']
        self.embInfo = embInfo
        # self.dense_num = len(self.dense_feas)  # 13 L
        self.sparse_num = len(embInfo)  # 26 C
        self.mode_type = mode_type
        self.embDim = embDim

        # embedding层，类别特征embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=self.embDim)
            for key, val in self.embInfo.items()
        })

        # crosspnn层
        self.crosspnn = CrossPnn(embDim, self.sparse_num, hiddenUnits, use_cuda)

        # dnn 层
        # hiddenUnits[0] += self.dense_num  # dense_inputs直接输入道dnn，没有embedding 256+13=269
        self.dnn_network = DNN(hiddenUnits, dnn_dropout)
        self.dense_final = nn.Linear(hiddenUnits[-1], 1)
        self.lossfunc = torch.nn.BCELoss()
        self.actifunc = torch.nn.Sigmoid()
        self.metricfunc = {"auc": get_auc}

    def forward(self, x, Debug=False, lossname='mae'):
        # features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'release_date', 'genre']
        inputs = x.long()
        inputs_embeds = []

        # inputs_embeds = [self.embed_layers['embed_' + key](inputs[:, i])
        #                  for key, i in zip(self.embInfo.keys(), range(inputs.shape[1]))]
        user_emb = self.embed_layers['embed_user_id'](inputs[:, 0])
        inputs_embeds.append(user_emb)
        movie_emb = self.embed_layers['embed_movie_id'](inputs[:, 1])
        inputs_embeds.append(movie_emb)
        inputs_embeds = torch.stack(inputs_embeds)  # [fea_num, batch_sz, embDim]->[7,128,10]
        # [None, sparse_num, embDim]  此时空间不连续， 下面改变形状不用view，用reshape
        inputs_embeds = inputs_embeds.permute((1, 0, 2))  # [batch_sz, fea_num, embDim]->[128,7,10]
        # crosspnn layer foward
        cross_outputs = self.crosspnn(inputs_embeds, inputs_embeds)
        l1 = F.relu(cross_outputs)
        # dnn_network
        dnn_x = self.dnn_network(l1)
        outputs = self.dense_final(dnn_x)
        outputs = self.actifunc(outputs)
        outputs = outputs.view(-1)
        return outputs


# -*- coding: utf-8 -*-
import datetime
import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable


class AutomaticWeightedLoss(torch.nn.Module):
    def __init__(self, num):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class TorchKerasModelCopy(torch.nn.Module):
    # print time bar...
    @staticmethod
    def print_bar():
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "=" * 80 + "%s" % nowtime)

    def __init__(self, net=None):
        super(TorchKerasModelCopy, self).__init__()
        self.net = net

    def forward(self, x):
        if self.net:
            return self.net.forward(x)
        else:
            raise NotImplementedError

    def compile(self, loss_func,
                optimizer=None, metrics_dict=None, device=None, idx=None):
        self.loss_func = loss_func
        self.optimizer = optimizer if optimizer else torch.optim.Adam(self.parameters(), lr=0.001)
        self.metrics_dict = metrics_dict if metrics_dict else {}
        self.history = {}
        self.device = device if torch.cuda.is_available() else None
        self.client_idx = idx
        if self.device:
            self.to(self.device)

    def train_step(self, features, labels, backward=True):

        self.train()
        if self.device:
            features = features.to(self.device)
            labels = labels.to(self.device)
        # forward
        predictions = self.forward(features)
        loss = self.loss_func(predictions, labels)
        # evaluate metrics
        train_metrics = {"loss": loss.item()}
        for name, metric_func in self.metrics_dict.items():
            train_metrics[name] = metric_func(predictions, labels).item()

        # backward
        if backward:
            self.optimizer.zero_grad()
            loss.backward()
            # update parameters
            self.optimizer.step()
        return train_metrics, loss

    @torch.no_grad()
    def evaluate_step(self, features, labels):

        self.eval()
        if self.device:
            features = features.to(self.device)
            labels = labels.to(self.device)
        with torch.no_grad():
            predictions = self.forward(features)
            loss = self.loss_func(predictions, labels)

        val_metrics = {"val_loss": loss.item()}
        for name, metric_func in self.metrics_dict.items():
            val_metrics["val_" + name] = metric_func(predictions, labels).item()

        return val_metrics

    def fitSDA(self, train_data, valid_data=None, onServer=True, log_step_freq=1000):

        self.history['cli'] = self.client_idx
        valid_data = valid_data if valid_data else []
        # 1，training loop -------------------------------------------------
        train_metrics_sum, step = {}, 0
        loss_epoch = 0
        for features, labels in train_data:
            if onServer:
                torch.index_select(features, dim=1, index=torch.tensor([0, 1, 5]))
            step = step + 1
            train_metrics, loss = self.train_step(features, labels)
            loss_epoch = loss_epoch + loss
            for name, metric in train_metrics.items():
                train_metrics_sum[name] = train_metrics_sum.get(name, 0.0) + metric

            if step % log_step_freq == 0:
                logs = {"step": step}
                logs.update({k: round(v / step, 3) for k, v in train_metrics_sum.items()})
                print(logs)
        loss_epoch = loss_epoch / step
        for name, metric_sum in train_metrics_sum.items():
            self.history[name] = self.history.get(name, []) + [metric_sum / step]

        # 2，validate loop -------------------------------------------------

        val_metrics_sum, step = {}, 0
        for features, labels in valid_data:
            step = step + 1
            val_metrics = self.evaluate_step(features, labels)
            for name, metric in val_metrics.items():
                val_metrics_sum[name] = val_metrics_sum.get(name, 0.0) + metric
        for name, metric_sum in val_metrics_sum.items():
            self.history[name] = self.history.get(name, []) + [metric_sum / step]

        return pd.DataFrame(self.history), loss_epoch

    @torch.no_grad()
    def evaluate(self, dl_quary):
        self.eval()
        val_metrics_list = {}
        for features, labels in dl_quary:
            val_metrics = self.evaluate_step(features, labels)
            for name, metric in val_metrics.items():
                val_metrics_list[name] = val_metrics_list.get(name, []) + [metric]

        return {name: np.mean(metric_list) for name, metric_list in val_metrics_list.items()}

    @torch.no_grad()
    def predict(self, dl):
        self.eval()
        if self.device:
            result = torch.cat([self.forward(t[0].to(self.device)) for t in dl])
        else:
            result = torch.cat([self.forward(t[0]) for t in dl])
        return (result.data)


def sda_init_net(cli_nums, fea_dims, modletype, use_cuda):
    global net
    emb_dim = 32
    hiddenUnits = [emb_dim * len(fea_dims), 32, 16, 8]
    nets = []
    for idx in range(cli_nums):
        lossname = 'auc'
        if modletype == 'mlp':
            net = MLP(fea_dims, hiddenUnits, lossname, emb_dim)
        elif modletype == 'metamf':
            net = MetaMF(fea_dims, lossname)
        elif modletype == 'pnn':
            net = PNN(fea_dims, hiddenUnits, use_cuda, lossname, emb_dim)
        elif modletype == 'dcn':
            net = DCN(fea_dims, hiddenUnits, 3, use_cuda, lossname, embDim=emb_dim)
        else:
            warnings.warn('modletype argument wrong', UserWarning)
        if use_cuda: net.cuda()
        net.apply(weights_init)
        net = TorchKerasModelCopy(net)
        nets.append(net)
    return nets


def eval_modle(nets, sDataValid, Valid, type=None):
    # 聚合之后测试模型
    auc_single = pd.DataFrame()
    auc_total = pd.DataFrame()
    for idx in range(len(nets)):
        net = nets[idx] if type == 'validate' else nets[idx][1]
        met_s = pd.DataFrame([net.evaluate(sDataValid[idx])])
        met_t = pd.DataFrame([net.evaluate(Valid)])
        auc_single = pd.concat([auc_single, met_s], axis=0)
        auc_total = pd.concat([auc_total, met_t], axis=0)
    print(" -------------------------------- mean on single data: ")
    print(auc_single.mean(axis=0))
    print(" -------------------------------- mean on all data: ")
    print(auc_total.mean(axis=0))
    return auc_single.mean(axis=0), auc_total.mean(axis=0)


def trainNet(Epoch, nets, modleType, sDataTrain, sDataValid, lr, best_nets, onServer):
    global train_auc
    for epo in range(1, Epoch + 1):
        if epo % 3 == 0:
            lr = float("%.4f" % (lr * np.power(0.94, 4)))
        # print("decay_lr: ", lr)
        train_auc = pd.DataFrame()
        for idx in range(len(nets)):
            # torch.nn.BCELoss()
            loss_func = nets[idx].net.lossfunc
            params = filter(lambda p: p.requires_grad, nets[idx].net.parameters())
            optimizer = torch.optim.Adam(params=params, lr=lr, weight_decay=0.001)
            metrics_dict = nets[idx].net.metricfunc
            cli_idx = idx + 1
            nets[idx].compile(loss_func=loss_func, optimizer=optimizer, metrics_dict=metrics_dict, idx=cli_idx)
            # 聚合之前support,前向传播
            met, loss = nets[idx].fitSDA(sDataTrain[idx], sDataValid[idx], onServer=onServer)
            train_auc = pd.concat([train_auc, met], axis=0)
            # 保存模型
            if epo % 1 == 0 and float("%.4f" % met['auc']) > best_nets[idx][0]:
                best_nets[idx][0] = float("%.4f" % met['auc'])
                best_nets[idx][1] = nets[idx]
        print(" ------------------------------------------------ matric modleType", modleType, epo)
        print(train_auc.mean(axis=0))
    return nets, best_nets, train_auc.mean(axis=0)


def setRequiresGradSomeTrue(nets, rating_col):
    for net in nets:
        for name, value in net.named_parameters():
            value.requires_grad = False
    for net in nets:
        for name, value in net.named_parameters():
            for clo in rating_col:
                if name.__contains__(clo):
                    value.requires_grad = True


def setRequiresGradAllTrue(nets):
    for net in nets:
        for name, value in net.named_parameters():
            value.requires_grad = True


def aggregateNet(nets, dData, aData, fed_type, use_cuda, strategy):
    # if t != Epoch:   # 联邦聚合 最后一轮不聚合
    # if fed_type == "Central":
    #     continue
    print(" -------------------------------- fed type:", fed_type)
    if fed_type == "FedAvg":
        average_all_layers(nets)
    if fed_type == "AggEmbPca":
        agg_user_emb(nets, dData, use_cuda, strategy['u'])
        agg_item_emb(nets, aData, use_cuda, strategy['v'])
    if fed_type == "AggEmbAvg":
        agg_user_emb(nets, dData, use_cuda, 'avg')
        agg_item_emb(nets, aData, use_cuda, 'avg')
        average_other_feas(nets)


def sda_fed_torchkeras(Epoch, nets, modleType, fed_type, strategy, sDataTrain, sDataValid, aData, dData, valid, test,
                       use_cuda):
    cli_nums = len(sDataTrain)
    lr = 0.001
    best_nets = {}
    for i in range(cli_nums):
        best_nets[i] = [-1, None]

    # 联邦训练
    for epo in range(1, Epoch + 1):
        # 载入服务器预训练模型参数后... 和用户无关的参数不更新
        aggregateNet(nets, dData, aData, fed_type, use_cuda, strategy)
        # 聚合之后测试模型
        print(" ------------------------------------------------ after agg:")
        eval_modle(nets, sDataValid, valid, 'validate')
        # 聚合之后训练模型， 只更新用户相关参数
        print(" ------------------------------------------------ device training, epo", epo)
        rating_col = ['user_id', 'movie_id', 'age', 'gender', 'occupation']
        setRequiresGradAllTrue(nets)
        nets, best_nets, train_auc = trainNet(1, nets, modleType, sDataTrain, sDataValid, lr, best_nets, onServer=False)

    # 测试模型
    print(" -------------------------------- fed done, test")
    auc_single, auc_total = eval_modle(best_nets, sDataValid, test, 'test')
    return auc_single, auc_total


def preTrain(Epoch, modleType, sDataTrain, sDataValid, test, fea_dims, use_cuda):
    # server 预训练
    lr = 0.001
    best_nets = {}
    cli_nums = len(sDataTrain)
    rating_col = ['user_id', 'movie_id', 'genre']
    train_fea_dims = {}
    for key, val in fea_dims.items():
        if key in rating_col:
            train_fea_dims[key] = val
    nets = sda_init_net(cli_nums, train_fea_dims, modleType, use_cuda)
    for i in range(cli_nums):
        best_nets[i] = [-1, None]

    _, best_nets, _ = trainNet(Epoch, nets, modleType, sDataTrain, sDataValid, lr, best_nets, onServer=True)

    # 测试模型
    print("\n ------------------------------------------------save pretrain test  modleType", modleType, '\n')
    _, _ = eval_modle(best_nets, sDataValid, test, 'test')
    nets = [best_nets[i][1] for i in range(len(best_nets))]
    for i in range(len(nets)):
        PATH = './' + modleType + str(i) + '.net'
        torch.save(nets[i], PATH)
    del best_nets, nets
    print("save net none...")


def loadPreTrain(modleType, sDataValid, test):
    # 读取预训练模型
    nets = []
    for i in range(len(sDataValid)):
        PATH = './' + modleType + str(i) + '.net'
        print('load net:', PATH)
        nets.append(torch.load(PATH))
    print("\n ------------------------------------------------load pretrain test modleType")
    _, _ = eval_modle(nets, sDataValid, test, 'validate')
    return nets


# 预训练
def train_sda():
    # use_cuda = torch.cuda.is_available()
    use_cuda = False
    print(use_cuda)
    type, path_pc, path_kg = 'single', '../dataset/preprocessed_data/ml-1m/single/', '../input/single-2/'
    # 读取数据
    fileS = ['s_0', 's_1', 's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9']
    fileA = ['a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'a_6', 'a_7', 'a_8', 'a_9',
             'a_10', 'a_11', 'a_12', 'a_13', 'a_14', 'a_15', 'a_16', 'a_17']
    # 读数据
    path, Epoch, test, auc_log_s, auc_log_t = path_pc, 2, True, {}, {}
    # path, Epoch, test, auc_log_s, auc_log_t = path_kg, 1, False, {}, {}
    sDataTrain, sDataValid, aData, dData, valid, test, fea_dims = readdata_SDA_ML1M(path, fileS, fileA, batch_size=512,
                                                                                     test=test)
    modles = ['mlp', 'pnn', 'dcn', 'metamf', 'mynet']
    fed_types = ["Central", "FedAvg", "AggEmbAvg", "AggEmbPca"]
    strategy = {'u': 'my_pca', 'v': 'my_pca'}  # ['avg', 'my_pca', 'pca']

    modleType = modles[0]
    preTrain(Epoch, modleType, sDataTrain, sDataValid, test, fea_dims, use_cuda)

    for modleType in modles[:1]:
        for fed_type in fed_types[1:3]:
            nets = loadPreTrain(modleType, sDataValid, test)
            auc_single, auc_total = sda_fed_torchkeras(Epoch, nets, modleType, fed_type, strategy, sDataTrain,
                                                       sDataValid, aData, dData, valid, test, use_cuda)
            name = modleType + fed_type
            auc_log_s[name] = auc_single
            auc_log_t[name] = auc_total

        print('\n ------------------------------------------------------------ mean auc_log on single\n')
        for name, val in auc_log_s.items():
            print("-----------", name, ":\n", val, '\n')
        print('\n ------------------------------------------------------------ mean auc_log on total\n')
        for name, val in auc_log_t.items():
            print("-----------", name, ":\n", val, '\n')


train_sda()

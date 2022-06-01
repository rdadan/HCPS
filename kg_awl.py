import matplotlib.pyplot as plt

"=================================================================================================================================================ulitiy"


def print_his(df, fed_type, lossname):
    if lossname == 'auc':
        cols = ['auc', 'val_auc', 'fed_auc']
        min_mae_idx = df[cols[0]].idxmax()
        min_mae = df[cols[0]].max()
        min_val_mae_idx = df[cols[1]].idxmax()
        min_val_mae = df[cols[1]].min()
        min_fed_mae_idx = df[cols[0]].idxmax()
        min_fed_mae = 0
        if fed_type not in ["Central", 'FedDomainCat']:
            min_fed_mae = df[cols[2]].max()
        print("\n%s_max_tra_idx: %d, %5.3f" % (lossname, min_mae_idx, min_mae))
        print("%s_max_val_idx: %d, %5.3f" % (lossname, min_val_mae_idx, min_val_mae))
        print("%s_max_fed_idx: %d, %5.3f" % (lossname, min_fed_mae_idx, min_fed_mae))
    elif lossname == 'mae':
        cols = ['mae', 'val_mae', 'fed_mae']
        min_mae_idx = df[cols[0]].idxmin()
        min_mae = df[cols[0]].min()
        min_val_mae_idx = df[cols[1]].idxmin()
        min_val_mae = df[cols[1]].min()
        min_fed_mae_idx = df[cols[0]].idxmin()
        min_fed_mae = 0
        if fed_type != "Central":
            min_fed_mae = df[cols[2]].min()
        print("\n%s_min_tra_idx: %d, %5.3f" % (lossname, min_mae_idx, min_mae))
        print("%s_min_val_idx: %d, %5.3f" % (lossname, min_val_mae_idx, min_val_mae))
        print("%s_min_fed_idx: %d, %5.3f" % (lossname, min_fed_mae_idx, min_fed_mae))


def plot_metric2(df, figname, lossname, cols=None, is_test=False):
    # epoch      loss       auc  val_loss   val_auc  fed_loss   fed_auc
    plt.title(figname)
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    if cols is None:
        if lossname == 'auc':
            cols = ['auc', 'val_auc', 'fed_auc']
        elif lossname == 'mae':
            cols = ['mae', 'val_mae', 'fed_mae']

    if figname.find("Central") != -1 or figname.find("Cat") != -1:
        cols = ['auc', 'val_auc']

    df = df[cols]
    plt.plot(df, label=cols)
    if is_test is False:
        print("save picture")
        picture = figname + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
        plt.savefig(picture, dpi=800)
    plt.legend()
    plt.show()


def average_all_layers(nets):
    state_dict = [x.state_dict() for x in nets]
    keys = list(state_dict[0].keys())
    for k in keys:
        if k.__contains__("emb") is False:
            weights = 0
            for i in range(len(nets)):
                weights = weights + state_dict[i][k]
            for net in nets:
                net.state_dict()[k].copy_(weights / len(nets))


def average_layers_without_emb(nets):
    state_dict = [x.state_dict() for x in nets]
    keys = list(state_dict[0].keys())
    for k in keys:
        if k.__contains__("emb") is False:
            weights = 0
            for i in range(len(nets)):
                weights = weights + state_dict[i][k]
            for net in nets:
                net.state_dict()[k].copy_(weights / len(nets))
    return nets


def get_avg_emb(nets, clis, fea, fea_val, emb_name, fea_val_label, use_cuda, same_dim):
    embs = []
    fea_labels = []
    for cli in clis:
        # fea_val->fea_lable
        fea_label = fea_val
        if same_dim is False:
            fea_label = fea_val_label[cli][fea][fea_val]
        fea_labels.append(fea_label)
        fea_emb = nets[cli].state_dict()[emb_name][fea_label]
        if use_cuda:
            fea_emb = fea_emb.cpu()
        fea_emb = fea_emb.numpy()
        embs.append(fea_emb)
    avg_emb = torch.mean(torch.tensor(embs), dim=0, keepdim=True).squeeze(0)
    return avg_emb, fea_labels


def update_userEmb_avg(nets, clientsdata, uv_cli_infos, fea_val_label, modletype, use_cuda, same_dim):
    # clientsdata columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre', 'rating']
    u_features = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    for uid, clis in uv_cli_infos[0].items():
        if len(clis) > 1:
            # 用户特征的val值
            u_fea_val = clientsdata[clis[0]][clientsdata[clis[0]]["user_id"] == uid].values[0][:5]
            # 特征emb的平均值
            for fea, fea_val in zip(u_features, u_fea_val):
                emb_name = "net.embed_layers.embed_" + fea + ".weight"
                if modletype == 'mlp':
                    emb_name = "net.embed_" + fea + ".weight"
                elif modletype == 'mynet':
                    emb_name = 'net.userEmb.weight'
                avg_emb, fea_labels = get_avg_emb(nets, clis, fea, fea_val, emb_name, fea_val_label, use_cuda, same_dim)
                # 更新
                for cli, fea_label in zip(clis, fea_labels):
                    nets[cli].state_dict()[emb_name][fea_label].copy_(avg_emb)
            # 只聚合user_id
            if modletype == 'mlp' or modletype == 'mynet':
                break


def update_itemEmb_avg(nets, clientsdata, uv_cli_infos, fea_val_label, modletype, use_cuda, same_dim):
    v_features = ['movie_id', 'genre']

    for vid, clis in uv_cli_infos[1].items():
        if len(clis) > 1:
            # item特征的lable值
            v_fea_values = clientsdata[clis[0]][clientsdata[clis[0]]["movie_id"] == vid].values[0][5:-1]
            # 特征emb的平均值
            for fea, fea_val in zip(v_features, v_fea_values):
                emb_name = "net.embed_layers.embed_" + fea + ".weight"
                if modletype == 'mlp':
                    emb_name = "net.embed_" + fea + ".weight"
                avg_emb, fea_labels = get_avg_emb(nets, clis, fea, fea_val, emb_name, fea_val_label, use_cuda, same_dim)
                # 更新
                for cli, fea_label in zip(clis, fea_labels):
                    nets[cli].state_dict()[emb_name][fea_label].copy_(avg_emb)
            # 只聚合user_id
            if modletype == 'mlp':
                break


def update_domain_emb_avg(nets, use_cuda, cross_type):
    embs_d = []
    name_d = 'net.shared_memory_d' if cross_type == 'crossmy' else 'net.domain.shared_memory_d'
    if cross_type == 'crossmy':
        name_d = 'net.shared_memory_d'
    elif cross_type == 'crossmeta':
        name_d = 'net.metaItemEmb.metaMemary'
    for net in nets:
        emb_d = net.state_dict()[name_d]
        if use_cuda:
            emb_d = emb_d.cpu()
        emb_d = emb_d.numpy()
        embs_d.append(emb_d)
    avg_emb_d = torch.mean(torch.tensor(embs_d), dim=0, keepdim=True).squeeze(0)
    for net in nets:
        net.state_dict()[name_d].copy_(avg_emb_d)


"=================================================================================================================================================readdata"

import itertools
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder


def get_fea_embInfo_same(data_all, features):
    fed_embInfos = {}
    for fea in features:
        fed_embInfos[fea] = data_all[fea].nunique()
    return fed_embInfos


def get_fea_embInfo_diff(cli):
    features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']
    # 每个客户端的emb div,以及特征值lable编码
    fea_val_lable_dict = {}
    # fea_emb_map = {'genre': 216}  # key:valus
    fea_embDim = {}
    for fea in features:
        # 每个特征的value值
        fea_value = cli[fea].unique()
        # 真实值转为label值
        le = LabelEncoder()
        cli[fea] = le.train_forward_transform(cli[fea])
        fea_label = cli[fea].unique()
        fea_val_lable_dict[fea] = dict(zip(fea_value, fea_label))
        fea_embDim[fea] = cli[fea].nunique()
    return cli, fea_val_lable_dict, fea_embDim


def create_dataloader(df, lossname, modle, use_cuda, batch_size=256):
    # 4以上转换为positive，其他转换为negative
    global tensor_x, tensor_y
    df_tensor = torch.tensor(df.values, dtype=torch.float32)
    if use_cuda:
        df_tensor = df_tensor.cuda()
    # if modle == 'pnn':
    #     tensor_x, tensor_y = torch.split(df_tensor, [7, 1], dim=1)
    if modle in ['mlp', 'pnn', 'dcn', 'metamf', 'mynet', 'mynet2']:
        tensor_x, tensor_y = torch.split(df_tensor, [2, 1], dim=1)
    else:
        import warnings
        warnings.warn('fed modle wrong', UserWarning)

    # 根据Tensor创建数据集
    dataset = TensorDataset(tensor_x, tensor_y.view(-1))
    # 使用DataLoader加载数据集
    train_data = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    return train_data  # , valid_data, test_data


def get_uv_cli_info(all_data, cli_data):
    # 用户/商品的分布
    uid_list = np.sort(all_data['user_id'].unique())
    vid_list = np.sort(all_data['movie_id'].unique())
    fea_uid_map = {}
    fea_vid_map = {}
    uv_data = []
    # 读取每个文件的数据
    for data in cli_data:
        uv_data.append([data["user_id"].unique(), data["movie_id"].unique()])
    # uid和vid的分布情况
    for (uid, vid) in itertools.zip_longest(uid_list, vid_list):
        idx = 0
        u_cli = []
        v_cli = []
        for cli in uv_data:
            if uid in cli[0]:
                u_cli.append(idx)
            if vid in cli[1]:
                v_cli.append(idx)
            idx += 1
        fea_uid_map[uid] = u_cli
        fea_vid_map[vid] = v_cli
    clo1 = ('user_id', 'u_clis')
    clo2 = ('movie_id', 'v_clis')
    df_u = pd.DataFrame(columns=clo1)
    df_v = pd.DataFrame(columns=clo2)

    df_u['user_id'] = [uid for uid in fea_uid_map.keys()]
    df_u['u_clis'] = [cli for cli in fea_uid_map.values()]
    df_v['movie_id'] = [vid for vid in fea_vid_map.keys()]
    df_v['v_clis'] = [cli for cli in fea_vid_map.values()]
    print("get uv_cli_info_done")
    return [df_u['u_clis'].to_dict(), df_v['v_clis'].to_dict()]


def read_uv_cli_info(path):
    # 读取用户分布
    import ast
    pa = path + 'u_cli_info.csv'
    u_cli = pd.read_csv(pa, converters={'u_clis': ast.literal_eval})
    fea_uid_map = {}
    for i in range(len(u_cli)):
        fea_uid_map[u_cli.loc[i]['user_id']] = u_cli.loc[i]['u_clis']
    # 商品的分布
    pa = path + 'v_cli_info.csv'
    v_cli = pd.read_csv(pa, converters={'v_clis': ast.literal_eval})
    fea_vid_map = {}
    for i in range(len(v_cli)):
        fea_vid_map[v_cli.loc[i]['movie_id']] = v_cli.loc[i]['v_clis']

    return [fea_uid_map, fea_vid_map]


def readdata_ML100k(path, train_step, valid_step, lossnames, modle, use_cuda, is_same_emb=True, istest=True, file=None):
    files = []
    if file is None:
        for idx in range(1, 6):
            train_file = path + "train_" + str(idx) + ".csv"
            files.append(train_file)
    else:
        for name in file:
            train_file = path + str(name) + ".csv"
            files.append(train_file)

    pa = path + "data_all.csv"
    all_data = pd.read_csv(pa)
    cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']
    if modle in ['mlp', 'pnn', 'dcn', 'metamf', 'mynet', 'mynet2']:
        cols = ["user_id", "movie_id", "rating"]
    all_data = all_data[cols[:-1]]
    # 各个客户端的训练数据
    train_datas = []
    # 列表存放每个客户端的验证集
    valid_datas = []
    # 全部客户端的数据
    total_valid_data = pd.DataFrame()
    # 列表存放每个客户端的验证集
    test_datas = []
    # 全部客户端的数据
    total_test_data = pd.DataFrame()
    # 存放全部客户端数据的列表
    client_data = []
    # 读取每个文件的数据
    fea_embInfo = []
    fea_value_label = []
    total_len = 0
    for idx in range(len(files)):
        columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre', 'rating']
        # 读取转换好格式的数据
        df = pd.read_csv(files[idx])  # , converters={'genre': literal_eval})
        df.columns = columns
        df = df[cols]
        if lossnames == "none":
            if idx % 2 == 0:
                lossname = 'auc'
            else:
                lossname = 'mae'
        else:
            lossname = lossnames

        if lossname == 'auc':
            df['rating'] = np.where(df['rating'] > 3, 1, 0)

        print("read file: ", files[idx], len(df))
        total_len += len(df)
        print("user_id: ", df["user_id"].nunique())

        df = df.apply(pd.to_numeric, errors='coerce')
        if istest:
            df = df[:5000]
        df_copy = copy.deepcopy(df)
        client_data.append(df_copy)
        if is_same_emb is False:
            df, fea_val_lable_dict, fea_embDim = get_fea_embInfo_diff(df)
            fea_value_label.append(fea_val_lable_dict)
            fea_embInfo.append(fea_embDim)
        # 分成训练集，验证集，测试集

        train_df = df.sample(frac=0.8, random_state=123, replace=False)
        other = df[~df.index.isin(train_df.index)]
        valid_df = other.sample(frac=0.5, random_state=123, replace=False)
        total_valid_data = total_valid_data.append(valid_df, ignore_index=True)
        test_df = other[~other.index.isin(valid_df.index)]
        total_test_data = total_test_data.append(test_df, ignore_index=True)
        # DataLoader加载
        batch_size = int(len(train_df) / train_step)
        print("batch_size: ", batch_size)
        train = create_dataloader(train_df, lossname, modle, use_cuda, batch_size)
        train_datas.append(train)
        batch_size = int(len(valid_df) / valid_step)
        valid = create_dataloader(valid_df, lossname, modle, use_cuda, batch_size)
        valid_datas.append(valid)
        batch_size = int(len(test_df) / valid_step)
        test = create_dataloader(test_df, lossname, modle, use_cuda, batch_size)
        test_datas.append(test)
    print("df total_len: ", total_len)
    total_valid_datas = create_dataloader(total_valid_data, lossname, modle, use_cuda)
    total_test_datas = create_dataloader(total_test_data, lossname, modle, use_cuda)
    if is_same_emb:
        fea_embInfo = get_fea_embInfo_same(all_data, cols[:-1])
    # uv_cli_info = read_uv_cli_info(path)
    uv_cli_info = get_uv_cli_info(all_data, client_data)
    return train_datas, valid_datas, test_datas, total_valid_datas, total_test_datas, fea_embInfo, uv_cli_info, client_data, fea_value_label


def readdata_ML100k_cnetral(path, lossname, modle, use_cuda, is_same_emb=True, istest=True):
    files = []
    for idx in range(1, 6):
        train_file = path + "train_" + str(idx) + ".csv"
        files.append(train_file)
    pa = path + "data_all.csv"
    all_data = pd.read_csv(pa)
    all_data = all_data.apply(pd.to_numeric, errors='coerce')
    if istest:
        all_data = all_data[:5000]
    # 分成训练集，验证集，测试集
    train = all_data.sample(frac=0.7, random_state=123, replace=False)
    other = all_data[~all_data.index.isin(train.index)]
    valid = other.sample(frac=0.4, random_state=123, replace=False)
    test = other[~other.index.isin(valid.index)]
    # DataLoader加载
    train_data = create_dataloader(train, lossname, modle, use_cuda)
    valid_data = create_dataloader(valid, lossname, modle, use_cuda)
    test_data = create_dataloader(test, lossname, modle, use_cuda)
    return train_data, valid_data, test_data


"=================================================================================================================================================net"
# def criterion(y_pred, y_true, log_vars):
#   loss = 0
#   for i in range(len(y_pred)):
#     precision = torch.exp(-log_vars[i])
#     diff = (y_pred[i]-y_true[i])**2. ## mse loss function
#     loss += torch.sum(precision * diff + log_vars[i], -1)
#   return torch.mean(loss)

import torch.nn as nn


class AutomaticWeightLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightLoss, self).__init__()
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


class MLP(nn.Module):
    def __init__(self, embInfo, hiddenUnits, lossname):
        super(MLP, self).__init__()
        self.userEmb = nn.Embedding(num_embeddings=embInfo['user_id'], embedding_dim=10)
        self.itemEmb = nn.Embedding(num_embeddings=embInfo['movie_id'], embedding_dim=10)
        """ fully connected layer """
        # self.MLP_Layers = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(Layers[:-1], Layers[1:]))])
        self.dnn_network = DNN(hiddenUnits)
        self.dense_final = nn.Linear(hiddenUnits[-1], 1)
        # dnn 层
        if lossname == 'mae':
            self.lossfunc = torch.nn.MSELoss()
            self.actifunc = torch.nn.ReLU()
            self.metricfunc = {"mae": get_mae}  # , "mse": get_mse, "rmse": get_rmse}

        elif lossname == 'auc':
            self.lossfunc = torch.nn.BCELoss()
            self.actifunc = torch.nn.Sigmoid()
            self.metricfunc = {"auc": get_auc}
        else:
            import warnings
            warnings.warn('loss argument wrong', UserWarning)

    def forward(self, inputs, lossname='mae'):
        """ 嵌入 """
        inputs = inputs.long()
        userEmb = self.userEmb(inputs[:, 0])
        itemEmb = self.itemEmb(inputs[:, 1])
        """ 拼接 """
        embedding_cat = torch.cat((userEmb, itemEmb), dim=1)
        # """ 点乘 """
        # embedding_vec = torch.mul(userEmb, itemEmb)
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
    def __init__(self, user_num, item_num, itemEmb_size=32, item_mem_num=8, userEmb_size=32, metaMemory=64,
                 hidden_size=64):  # note that we have many users and each user has many layers
        super(MetaRecommender, self).__init__()
        self.item_num = item_num
        self.itemEmb_size = itemEmb_size
        self.item_mem_num = item_mem_num
        # For each user
        self.userEmbedding = nn.Embedding(user_num, userEmb_size)
        self.memory = torch.nn.Parameter(nn.init.xavier_normal_(torch.Tensor(userEmb_size, metaMemory)),
                                         requires_grad=True)
        # For each layer
        self.hidden_layer_1, self.weight_layer_1, self.bias_layer_1 = self.define_one_layer(metaMemory, hidden_size,
                                                                                            itemEmb_size,
                                                                                            int(itemEmb_size / 4))
        self.hidden_layer_2, self.weight_layer_2, self.bias_layer_2 = self.define_one_layer(metaMemory, hidden_size,
                                                                                            int(itemEmb_size / 4), 1)
        self.hidden_layer_3, self.emb_layer_1, self.emb_layer_2 = self.define_itemEmbedding(item_num, itemEmb_size,
                                                                                            item_mem_num, metaMemory,
                                                                                            hidden_size)

    def define_one_layer(self, metaMemory, hidden_size, int_size, out_size):  # define one layer in MetaMF
        hidden_layer = nn.Linear(metaMemory, hidden_size)
        weight_layer = nn.Linear(hidden_size, int_size * out_size)
        bias_layer = nn.Linear(hidden_size, out_size)
        return hidden_layer, weight_layer, bias_layer

    def define_itemEmbedding(self, item_num, itemEmb_size, item_mem_num, metaMemory, hidden_size):
        hidden_layer = nn.Linear(metaMemory, hidden_size)
        emb_layer_1 = nn.Linear(hidden_size, item_num * item_mem_num)
        emb_layer_2 = nn.Linear(hidden_size, item_mem_num * itemEmb_size)
        return hidden_layer, emb_layer_1, emb_layer_2

    def forward(self, user_id):
        # collaborative memory module
        userEmb = self.userEmbedding(user_id)  # input_user=[batch_size, userEmb_size]
        cf_vec = torch.matmul(userEmb,
                              self.memory)  # [user_num, u_emb_size]*[u_emb_size, metaMemory]=[batch_size, metaMemory]
        output_weight = []
        output_bias = []
        weight, bias = self.get_one_layer(self.hidden_layer_1, self.weight_layer_1, self.bias_layer_1, cf_vec,
                                          self.itemEmb_size, int(self.itemEmb_size / 4))
        output_weight.append(weight)
        output_bias.append(bias)
        weight, bias = self.get_one_layer(self.hidden_layer_2, self.weight_layer_2, self.bias_layer_2, cf_vec,
                                          int(self.itemEmb_size / 4), 1)
        output_weight.append(weight)
        output_bias.append(bias)
        itemEmbedding = self.get_itemEmbedding(self.hidden_layer_3, self.emb_layer_1, self.emb_layer_2, cf_vec,
                                               self.item_num, self.item_mem_num, self.itemEmb_size)
        # meta recommender module
        return output_weight, output_bias, itemEmbedding, cf_vec  # ([len(layer_list)+1, batch_size, *, *], [len(layer_list)+1, batch_size, 1, *], [batch_size, item_num, itemEmb_size], [batch_size, metaMemory])

    def get_one_layer(self, hidden_layer, weight_layer, bias_layer, cf_vec, int_size,
                      out_size):  # get one layer in MetaMF
        hid = hidden_layer(cf_vec)  # hid=[batch_size, hidden_size]
        hid = F.relu(hid)
        weight = weight_layer(hid)  # weight=[batch_size, self.layer_list[i-1]*self.layer_list[i]]
        bias = bias_layer(hid)  # bias=[batch_size, self.layer_list[i]]
        weight = weight.view(-1, int_size, out_size)
        bias = bias.view(-1, 1, out_size)
        return weight, bias

    def get_itemEmbedding(self, hidden_layer, emb_layer_1, emb_layer_2, cf_vec, item_num, item_mem_num, itemEmb_size):
        hid = hidden_layer(cf_vec)  # hid=[batch_size, hidden_size]
        hid = F.relu(hid)
        emb_left = emb_layer_1(hid)  # emb_left=[batch_size, item_num*item_mem_num]
        emb_right = emb_layer_2(hid)  # emb_right=[batch_size, item_mem_num*itemEmb_size]
        emb_left = emb_left.view(-1, item_num, item_mem_num)  # emb_left=[batch_size, item_num, item_mem_num]
        emb_right = emb_right.view(-1, item_mem_num,
                                   itemEmb_size)  # emb_right=[batch_size, item_mem_num, itemEmb_size]
        itemEmbedding = torch.matmul(emb_left, emb_right)  # itemEmbedding=[batch_size, item_num, itemEmb_size]
        return itemEmbedding


class MetaMF(nn.Module):
    def __init__(self, embInfo, lossname, itemEmb_size=10, item_mem_num=8, userEmb_size=10, metaMemory=64,
                 hidden_size=64):
        super(MetaMF, self).__init__()
        self.item_num = embInfo['movie_id']
        self.metarecommender = MetaRecommender(embInfo['user_id'], embInfo['movie_id'], itemEmb_size,
                                               item_mem_num, userEmb_size, metaMemory, hidden_size)
        self.lossname = lossname
        if lossname == 'mae':
            self.lossfunc = torch.nn.MSELoss()
            self.actifunc = torch.nn.ReLU()
            self.metricfunc = {"mae": get_mae}  # , "mse": get_mse, "rmse": get_rmse}

        elif lossname == 'auc':
            self.lossfunc = torch.nn.BCELoss()
            self.actifunc = torch.nn.Sigmoid()
            self.metricfunc = {"auc": get_auc}

    def forward(self, inputs):
        # prediction module
        inputs = inputs.long()
        TorchKerasModel_weight, TorchKerasModel_bias, itemEmbedding, _ = self.metarecommender(inputs[:, 0])
        item_id = inputs[:, 1]
        item_id = item_id.view(-1, 1)  # item_id=[batch_size, 1]
        item_one_hot = torch.zeros(len(item_id), self.item_num,
                                   device=item_id.device)  # we generate it dynamically, and default device is cpu
        item_one_hot.scatter_(1, item_id, 1)  # item_one_hot=[batch_size, item_num]
        item_one_hot = torch.unsqueeze(item_one_hot, 1)  # item_one_hot=[batch_size, 1, item_num]
        itemEmb = torch.matmul(item_one_hot, itemEmbedding)  # out=[batch_size, 1, itemEmb_size]
        out = torch.matmul(itemEmb, TorchKerasModel_weight[0])  # out=[batch_size, 1, itemEmb_size/4]
        out = out + TorchKerasModel_bias[0]  # out=[batch_size, 1, itemEmb_size/4]
        out = F.relu(out)  # out=[batch_size, 1, itemEmb_size/4]
        out = torch.matmul(out, TorchKerasModel_weight[1])  # out=[batch_size, 1, 1]
        out = out + TorchKerasModel_bias[1]  # out=[batch_size, 1, 1]
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
    def __init__(self, embInfo, hiddenUnits, layer_num, use_cuda, lossname, embDim=10):
        super(DCN, self).__init__()
        # self.dense_embInfo, self.embInfo = embInfo
        self.embInfo = embInfo
        # embedding层，类别特征embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=embDim)
            for key, val in embInfo.items()
        })

        hiddenUnits.insert(0, len(embInfo) * embDim)
        self.cross_network = CrossDcn(layer_num, hiddenUnits[0])  # layer_num是交叉网络的层数， hiddenUnits[0]表示输入的整体维度大小
        self.dnn_network = DNN(hiddenUnits)
        self.dense_final = nn.Linear(hiddenUnits[-1] + hiddenUnits[0], 1)
        self.lossname = lossname
        if lossname == 'mae':
            self.lossfunc = torch.nn.MSELoss()
            self.actifunc = torch.nn.ReLU()
            self.metricfunc = {"mae": get_mae}  # , "mse": get_mse, "rmse": get_rmse}

        elif lossname == 'auc':
            self.lossfunc = torch.nn.BCELoss()
            self.actifunc = torch.nn.Sigmoid()
            self.metricfunc = {"auc": get_auc}
        else:
            import warnings
            warnings.warn('loss argument wrong', UserWarning)

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
    def __init__(self, embInfo, hiddenUnits, use_cuda, lossname, mode_type='in', dnn_dropout=0.3, embDim=10):
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
        if lossname == 'mae':
            self.lossfunc = torch.nn.MSELoss()
            self.actifunc = torch.nn.ReLU()
            self.metricfunc = {"mae": get_mae}  # , "mse": get_mse, "rmse": get_rmse}

        elif lossname == 'auc':
            self.lossfunc = torch.nn.BCELoss()
            self.actifunc = torch.nn.Sigmoid()
            self.metricfunc = {"auc": get_auc}
        else:
            import warnings
            warnings.warn('loss argument wrong', UserWarning)

    def forward(self, x, Debug=False, lossname='mae'):
        # features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'release_date', 'genre']
        inputs = x.long()
        inputs_embeds = []

        # inputs_embeds = [self.embed_layers['embed_' + key](inputs[:, i])
        #                  for key, i in zip(self.embInfo.keys(), range(inputs.shape[1]))]
        userEmb = self.embed_layers['userEmb'](inputs[:, 0])
        inputs_embeds.append(userEmb)
        movie_emb = self.embed_layers['itemEmb'](inputs[:, 1])
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


def get_uv_embedding(emb_layer_1, emb_layer_2, cf_vec, domain_num, metaMemory_num, domain_emb_size, droprate):
    emb_left = emb_layer_1(cf_vec)
    emb_right = emb_layer_2(cf_vec)
    emb_left = emb_left.view(-1, domain_num, metaMemory_num)  # emb_left=[batch_size, domain_num, metaMemory_num]
    emb_right = emb_right.view(-1, metaMemory_num,
                               domain_emb_size)  # emb_right=[batch_size, metaMemory_num, domain_emb_size]
    private_domain_emb = torch.matmul(emb_left, emb_right)
    return private_domain_emb


def set_uv_emb_layer(domain_num, embDim, mem_num, metaMemory):
    emb_layer_1 = nn.Linear(metaMemory, domain_num * mem_num)
    emb_layer_2 = nn.Linear(metaMemory, mem_num * embDim)
    return emb_layer_1, emb_layer_2


class DomainLayer(nn.Module):  # in fact, it's not a hypernetwork
    def __init__(self, domain_num, embDim, mem_num, memNum, droprate):
        super(DomainLayer, self).__init__()
        self.domain_num = domain_num
        self.embDim = embDim
        self.mem_num = mem_num
        self.droprate = droprate
        self.domain_embedding = nn.Embedding(domain_num, embDim)
        self.shared_memory_d = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(embDim, memNum)), requires_grad=True)
        self.embLayer1, self.embLayer2 = set_uv_emb_layer(domain_num, embDim, mem_num, memNum)

    def forward(self, domain_id):
        domain_emb = self.domain_embedding(domain_id)
        domain_cf_vec_d = torch.matmul(domain_emb, self.shared_memory_d)  # [1,embDim]*[embDim,memNum]=[1,memNum]
        d_embs = get_uv_embedding(self.embLayer1, self.embLayer2, domain_cf_vec_d,
                                  self.domain_num, self.mem_num, self.embDim, self.droprate)  # 【1，18，10】
        return d_embs.squeeze(0)


class MetaAggregator(nn.Module):  # in fact, it's not a hypernetwork
    def __init__(self, embDim, metaMemry):
        super(MetaAggregator, self).__init__()
        self.embDim = embDim
        self.metaMemary = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(embDim, metaMemry)))
        self.embLayer = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(metaMemry, embDim)))

    def forward(self, emb):
        embMemary = torch.matmul(emb, self.metaMemary)  # [batch,embDim]*[batch,metaMemry]=[batch,memNum]
        metaEmb = torch.matmul(embMemary, self.embLayer)
        return metaEmb


class myNet(nn.Module):
    def __init__(self, embInfo, domain_id, uv_domains_info, hiddenUnits, use_cuda, lossname, cross_type,
                 dropout=0.0, embDim=32, metaMemory=32, memNum=8):
        super(myNet, self).__init__()
        # self.features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']
        hiddenUnits = [64, 64, 16, 8]
        self.use_cuda = use_cuda
        self.embInfo = embInfo
        self.cross_type = cross_type
        self.domain_id = domain_id
        self.uv_domains_info = uv_domains_info
        self.userEmb = nn.Embedding(num_embeddings=embInfo['user_id'], embedding_dim=embDim)
        self.itemEmb = nn.Embedding(num_embeddings=embInfo['movie_id'], embedding_dim=embDim)
        if cross_type == 'crossmy':
            # embDim = embDim * 2
            # self.private_domain = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(1, metaMemory)))
            self.shared_memory_d = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(embDim, metaMemory)))
            self.domain_weights_u = nn.Parameter(torch.rand(embDim, 1))
            self.domain_bias_u = nn.Parameter(torch.rand(embDim, 1))
            self.domain_weights_v = nn.Parameter(torch.rand(embDim, 1))
            self.domain_bias_v = nn.Parameter(torch.rand(embDim, 1))
            # self.losswi = nn.Parameter(torch.rand(1, 1))
        elif cross_type == 'crossmeta':
            self.metaItemEmb = MetaAggregator(embDim, metaMemory)
        elif cross_type == 'crosspnn':
            self.domain_embedding = nn.Embedding(embInfo['domain_id'], embDim)
            self.domain = DomainLayer(embInfo['domain_id'], embDim, memNum, metaMemory, dropout)
            self.mycross = CrossPnn(embDim, 4, hiddenUnits, use_cuda)  # sparse_num=4
        elif cross_type == 'crossdcn':
            self.domain_embedding = nn.Embedding(embInfo['domain_id'], embDim)
            self.domain = DomainLayer(embInfo['domain_id'], embDim, memNum, metaMemory, dropout)
            self.mycross = CrossDcn(3, hiddenUnits[0])  # cross_layer_num=3
        # dnn 层
        # self.dnn_network = DNN(hiddenUnits, dropout)
        self.dense_final = nn.Linear(hiddenUnits[0], 1)
        if lossname == 'mae':
            self.lossname = lossname
            self.lossfunc = torch.nn.MSELoss()
            self.actifunc = torch.nn.ReLU()
            self.metricfunc = {"mae": get_mae}  # , "mse": get_mse, "rmse": get_rmse}
        elif lossname == 'auc':
            self.lossname = lossname
            self.lossfunc = torch.nn.BCELoss()
            self.actifunc = torch.nn.Sigmoid()
            self.metricfunc = {"auc": get_auc}
        else:
            import warnings
            warnings.warn('loss argument wrong', UserWarning)

    def forward(self, inputs):
        global l1
        inputs = inputs.long()
        userEmb = self.userEmb(inputs[:, 0])
        itemEmb = self.itemEmb(inputs[:, 1])
        # 域
        if self.cross_type == 'crossmy':
            # share_domain = torch.matmul(self.private_domain, self.shared_memory_d)
            emb_u = torch.unsqueeze(userEmb, dim=2)
            emb_v = torch.unsqueeze(itemEmb, dim=2)
            # metaU = torch.matmul(torch.matmul(userEmb, self.shared_memory_d), self.domain_weights_u) + self.domain_bias_u
            # meta_v = torch.matmul(torch.matmul(itemEmb, self.shared_memory_d), self.domain_weights_v) + self.domain_bias_v
            metaU = torch.matmul(userEmb, self.shared_memory_d)
            meta_v = torch.matmul(itemEmb, self.shared_memory_d)
            metaEmb = torch.cat((metaU, meta_v), 1)
            # metaEmb = F.relu(metaEmb)
            l1 = torch.squeeze(metaEmb)
        elif self.cross_type == 'crossmeta':
            metaV = self.metaItemEmb(itemEmb)
            metaEmb = torch.cat((userEmb, metaV), 1)
            l1 = torch.squeeze(metaEmb)
        elif self.cross_type == 'crosscat':
            l1 = torch.cat((userEmb, itemEmb), 1)
        else:
            inputs_embeds = []
            private_domain_space = self.domain(self.domain_id)
            # at user space
            user_domain_avgemb = self.get_uv_domain_avgemb(private_domain_space, inputs[:, 0], self.uv_domains_info[0])
            item_domain_avgemb = self.get_uv_domain_avgemb(private_domain_space, inputs[:, 1], self.uv_domains_info[1])
            inputs_embeds.append(user_domain_avgemb)
            inputs_embeds.append(item_domain_avgemb)
            if self.cross_type == 'crossdcn':
                embeds = torch.cat(inputs_embeds, 1)  # [128, 10*4]
                l1 = self.mycross(embeds)
            elif self.cross_type == 'crosspnn':
                embeds = torch.stack(inputs_embeds).permute((1, 0, 2))  # [4,128,10]->[128,4,10]
                out = self.mycross(embeds, embeds)
                l1 = F.relu(out)  # [128，64]
        # l1 = self.dnn_network(l1)
        outputs = self.dense_final(l1)
        outputs = self.actifunc(outputs)
        outputs = outputs.view(-1)
        return outputs


class myNet2(nn.Module):
    def __init__(self, embInfo, cliNum, uv_domains_info, hiddenUnits, use_cuda, lossname, cross_type,
                 dropout=0.0, embDim=16, metaMemory=32, memNum=8):
        super(myNet2, self).__init__()
        hiddenUnits = [32, 32, 16, 8]
        self.use_cuda = use_cuda
        self.embInfo = embInfo
        self.cliNum = cliNum
        self.cross_type = cross_type
        self.uv_domains_info = uv_domains_info
        userEmbName, itemEmbName = [], []
        for idx in range(cliNum):
            userEmbName.append("userEmb" + "Cli_" + str(idx))
            itemEmbName.append("itemEmb" + "Cli_" + str(idx))

        self.userEmbLayers = nn.ModuleDict({
            embName: nn.Embedding(num_embeddings=embInfo['user_id'], embedding_dim=embDim)
            for embName in userEmbName
        })
        self.itemEmbLayers = nn.ModuleDict({
            embName: nn.Embedding(num_embeddings=embInfo['movie_id'], embedding_dim=embDim)
            for embName in itemEmbName
        })
        self.metaItemEmb = MetaAggregator(embDim, metaMemory)
        self.dense_final = nn.Linear(hiddenUnits[0], 1)
        if lossname == 'auc':
            self.lossname = lossname
            self.lossfunc = torch.nn.BCELoss()
            self.actifunc = torch.nn.Sigmoid()
            self.metricfunc = {"auc": get_auc}
        else:
            import warnings
            warnings.warn('loss argument wrong', UserWarning)

    def forward(self, inputs):
        outputs = []
        # inputs_embeds = [self.embed_layers['embed_' + key](inputs[:, i])
        #                  for key, i in zip(self.embInfo.keys(), range(inputs.shape[1]))]

        for input, userEmb, itemEmb in zip(inputs, self.userEmbLayers.values(), self.itemEmbLayers.values()):
            input = input.long()
            userEmb = userEmb(input[:, 0])
            itemEmb = itemEmb(input[:, 1])
            metaV = self.metaItemEmb(itemEmb)
            catEmb = torch.cat((userEmb, metaV), 1)
            catEmb = torch.squeeze(catEmb)
            output = self.dense_final(catEmb)
            output = self.actifunc(output)
            output = output.view(-1)
            outputs.append(output)
        return outputs

    def get_emb(self, id, num, emb):
        id = id.view(-1, 1)
        one_hot = torch.zeros(len(id), num, device=id.device)  # we generate it dynamically, and default device is cpu
        one_hot.scatter_(1, id, 1)  # item_one_hot=[batch_size, domain_num]
        one_hot = torch.unsqueeze(one_hot, 1)  # item_one_hot=[batch_size, 1, domain_num]
        embed = torch.matmul(one_hot, emb)  # out=[batch_size, 1, domain_emb_size]
        embed = embed.squeeze()
        return embed

    def get_uv_domain_avgemb(self, private_domain_space, input_ids, uv_domains):
        domain_avgemb = []
        input_ids = input_ids.data.detach().cpu().numpy()
        for id in input_ids:
            domains = uv_domains[id]
            embs = []
            for d in domains:
                if self.use_cuda:
                    embs.append(private_domain_space[d].detach().cpu().numpy())
                else:
                    embs.append(private_domain_space[d].detach().numpy())
            avg_emb = torch.mean(torch.tensor(embs), dim=0, keepdim=True).squeeze(0)
            domain_avgemb.append(avg_emb)

        domain_avgemb = torch.stack(domain_avgemb, 0)
        if self.use_cuda:
            domain_avgemb = domain_avgemb.cuda()
        return domain_avgemb


"————————————————————————————————————————local train/test——————————————————————————————————————————"

import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy
from sklearn.metrics import roc_auc_score


def loss_bcewithlogit():  # Cross Entropy Loss
    return torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()


def loss_mse():
    return torch.nn.MSELoss()


def get_mae_mse_rmse(y_true, y_pred):
    # y_true = y_true.cuda().data.cpu()
    # y_pred = y_pred.data.cpu().numpy()
    y_pred = y_pred.data.detach().cpu().numpy()
    y_true = y_true.data.detach().cpu().numpy()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # ** 0.5

    return mae, mse, rmse


def get_mae(y_pred, y_true):
    y_pred = y_pred.data.detach().cpu().numpy()
    y_true = y_true.data.detach().cpu().numpy()
    mae = mean_absolute_error(y_true, y_pred)
    return mae


def get_mse(y_pred, y_true):
    y_pred = y_pred.data.detach().cpu().numpy()
    y_true = y_true.data.detach().cpu().numpy()
    mse = mean_squared_error(y_true, y_pred)
    return mse


def get_rmse(y_pred, y_true):
    y_pred = y_pred.data.detach().cpu().numpy()
    y_true = y_true.data.detach().cpu().numpy()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # ** 0.5
    return rmse


# 计算AUC
def get_auc(y_pred, y_true):
    y_pred = y_pred.data.detach().cpu().numpy()
    y_true = y_true.data.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)


def get_aucs(y_preds, y_trues):
    aucs = []
    for y_pred, y_true in zip(y_preds, y_trues):
        y_pred = y_pred.data.detach().cpu().numpy()
        y_true = y_true.data.detach().cpu().numpy()

        aucs.append(roc_auc_score(y_true, y_pred))
    return aucs


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


# -*- coding: utf-8 -*-
import datetime
import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable


class AutomaticWeightLoss(torch.nn.Module):
    def __init__(self, num):
        super(AutomaticWeightLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class TorchKerasModel(torch.nn.Module):
    # print time bar...
    @staticmethod
    def print_bar():
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "=" * 80 + "%s" % nowtime)

    def __init__(self, net=None):
        super(TorchKerasModel, self).__init__()
        self.net = net

    def forward(self, x):
        if self.net:
            return self.net.forward(x)
        else:
            raise NotImplementedError

    def compile(self, loss_func,
                optimizer=None, metrics_dict=None, device=None):
        self.loss_func = loss_func
        self.optimizer = optimizer if optimizer else torch.optim.Adam(self.parameters(), lr=0.001)
        self.metrics_dict = metrics_dict if metrics_dict else {}
        self.device = device if torch.cuda.is_available() else None
        self.train_step = 1
        self.history = {}
        self.train_metrics_sum = {}
        self.device = device if torch.cuda.is_available() else None
        if self.device:
            self.to(self.device)

    def get_train_Loss(self, features, labels):
        self.train()
        self.optimizer.zero_grad()
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

        return loss, train_metrics

    @torch.no_grad()
    def evaluate_train_step(self, features, labels):

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

    def my_train_step(self, features, labels, awl=None):

        self.train()
        self.optimizer.zero_grad()
        if self.device:
            features = features.to(self.device)
            labels = labels.to(self.device)

        # forward
        predictions = self.forward(features)
        losses = []
        train_metrics = []
        for prediction, label in zip(predictions, labels):
            metric = {}
            loss = self.loss_func(prediction, label)
            metric['loss'] = loss.item()
            for name, metric_func in self.metrics_dict.items():
                metric[name] = metric_func(prediction, label).item()
            losses.append(loss)
            train_metrics.append(metric)

        # # backward
        if awl:
            loss = awl(losses)
        loss.backward()
        # # update parameters
        self.optimizer.step()
        self.optimizer.zero_grad()

        return train_metrics

    def my_valid_train_step(self, features, labels):
        self.eval()
        if self.device:
            features = features.to(self.device)
            labels = labels.to(self.device)
        with torch.no_grad():
            losses = []
            valid_metrics = []
            predictions = self.forward(features)
            for prediction, label in zip(predictions, labels):
                metric = {}
                loss = self.loss_func(prediction, label)
                metric['val_loss'] = loss.item()
                for name, metric_func in self.metrics_dict.items():
                    name = "val_"+name
                    metric[name] = metric_func(prediction, label).item()
                losses.append(loss)
                valid_metrics.append(metric)
        return valid_metrics

    def get_train_step_data(self, current_train_step, train_datas):
        # 1，training train_step loop ------------------------------------------------
        features, labels = [], []
        for train_data in train_datas:
            for train_step, (feature, label) in enumerate(train_data, 1):
                if train_step == current_train_step:
                    features.append(feature)
                    labels.append(label)
                    break
        return features, labels

    def train_forward(self, cli_idx, train_train_step, dl_train, train_step):
        global LOSS
        # 1，training train_step loop ------------------------------------------------
        for train_step, (features, labels) in enumerate(dl_train, 1):
            if train_step == train_train_step:
                LOSS, train_metrics = self.get_train_Loss(features, labels)
                for name, metric in train_metrics.items():
                    self.train_metrics_sum[name] = self.train_metrics_sum.get(name, 0.0) + metric
                self.train_step += 1
                if train_step % 50 == 0 and train_step != train_step:
                    logs = {"cli:": cli_idx + 1, "train_step": train_step}
                    logs.update({k: round(v / train_step, 4) for k, v in self.train_metrics_sum.items()})
                    print(logs)
                break
        return LOSS

    def train_metrics(self, cli_idx, epoch, train_train_step, dl_val=None):

        # 1. train_metrics_sum
        for name, metric_sum in self.train_metrics_sum.items():
            self.history[name] = self.history.get(name, []) + [metric_sum / train_train_step]

        # 2 val_metrics_sum
        val_metrics_sum, val_train_step = {}, 0
        for features, labels in dl_val:
            val_train_step = val_train_step + 1
            val_metrics = self.evaluate_train_step(features, labels)
            for name, metric in val_metrics.items():
                val_metrics_sum[name] = val_metrics_sum.get(name, 0.0) + metric
        for name, metric_sum in val_metrics_sum.items():
            self.history[name] = self.history.get(name, []) + [metric_sum / val_train_step]

        # 3 print logs
        infos = {"cli": cli_idx + 1, "epoch": epoch + 1, "train_step": train_train_step}
        infos.update({k: round(self.history[k][-1], 4) for k in self.history})
        tb = PrettyTable()
        tb.field_names = infos.keys()
        tb.add_row(infos.values())
        print("\n", tb)
        # TorchKerasModel.print_bar()
        return LOSS  # , pd.DataFrame(self.history)

    @torch.no_grad()
    def evaluate(self, dl_val):
        self.eval()
        val_metrics_list = {}
        for features, labels in dl_val:
            val_metrics = self.evaluate_train_step(features, labels)
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


# torchkeras
def init_net(cli_nums, uv_cli_info, use_cuda, same_dim, fea_embInfo, modletype, lossnames, cross_type):
    global net

    hiddenUnits = [64, 32, 16]
    if cross_type == 'torchcat': hiddenUnits = [20, 64, 32, 16]
    local_nets = []
    # 模型初始化
    for idx in range(cli_nums):
        if lossnames == "none":
            if idx % 2 == 0:
                lossname = 'auc'
            else:
                lossname = 'mae'
        else:
            lossname = lossnames
        print(modletype)
        embInfo = fea_embInfo if same_dim else fea_embInfo[idx]
        if modletype == 'mlp':
            hiddenUnits = [20, 64, 32, 16]
            net = MLP(embInfo, hiddenUnits, lossname)
        elif modletype == 'metamf':
            net = MetaMF(embInfo, lossname)
        elif modletype == 'pnn':
            net = PNN(embInfo, hiddenUnits, use_cuda, lossname)
        elif modletype == 'dcn':
            net = DCN(embInfo, hiddenUnits, 3, use_cuda, lossname)
        elif modletype == 'mynet':
            domain_id = torch.tensor(idx).long()
            if use_cuda: domain_id = domain_id.cuda()
            embInfo['domain_id'] = cli_nums
            hiddenUnits = [20, 64, 32, 16]
        elif modletype == 'mynet2':
            net = myNet2(embInfo, cli_nums, uv_cli_info, hiddenUnits, use_cuda, lossname, cross_type)
        else:
            import warnings
            warnings.warn('modletype argument wrong', UserWarning)

        if use_cuda: net.cuda()
        net.apply(weights_init)
        # if modletype == 'mynet2':
        #     import torchkeras
        #     net = torchkeras.Model(net)
        #     local_nets.append(net)
        #     break
        # else:
        net = TorchKerasModel(net)
        local_nets.append(net)
        if modletype == 'mynet2':
            break
    return local_nets


def fed_torchkeras(Epochs, train_step, fed_type, train_datas, valid_datas, test_datas, total_valid_datas, total_test_datas,
                   fea_embInfo,
                   uv_cli_info, client_data, fea_val_label, use_cuda, modletype, same_dim, test_type, lossnames,
                   cross_type):
    cli_nums = len(train_datas)
    local_nets = init_net(cli_nums, uv_cli_info, use_cuda, same_dim, fea_embInfo, modletype, lossnames, cross_type)
    # 全部客户端的metrics记录格式
    # cols = ["epoch", "loss", "auc", "val_loss", "val_auc", "fed_loss", "fed_auc"]
    # tra_met, val_met, fed_met = 'auc', 'val_auc', 'fed_auc'
    cols_auc = ["epoch", "loss", 'auc', "val_loss", 'val_auc', "fed_loss", 'fed_auc']
    # tra_met, val_met, fed_met = 'mae', 'val_mae', 'fed_mae'
    cols_mae = ["epoch", "loss", 'mae', "val_loss", 'val_mae', "fed_loss", 'fed_mae']

    if fed_type in ['Central', 'FedDomainCat']:
        cols_auc = cols_auc[:-2]
        cols_mae = cols_mae[:-2]
    cols_cli = []
    best_net = {}
    for i in range(cli_nums):
        cols_cli.append("train_cli_" + str(i))
        cols_cli.append("fed_cli_" + str(i))
        best_net[i] = [99, None]

    # 联邦训练
    import itertools
    awl = AutomaticWeightLoss(cli_nums)
    awl_optimizer = torch.optim.Adam(params=awl.parameters(), lr=0.01)
    lr = 0.001
    for epoch in range(Epochs):
        if (epoch + 1) % 20 == 0:
            lr = float("%.4f" % (lr * np.power(0.94, 4)))
            # lr = round(lr * np.power(0.97, (T / 3)), 4)
        print('\n', modletype, fed_type, cross_type, "decay_lr: ", lr, "epoch ", epoch + 1)
        # train
        for idx in range(cli_nums):
            lossfunc = local_nets[idx].net.lossfunc
            metricfunc = local_nets[idx].net.metricfunc
            params = local_nets[idx].net.parameters()
            optimizer = torch.optim.Adam(params=params, lr=lr, weight_decay=0.001)
            local_nets[idx].compile(lossfunc, optimizer, metricfunc)
        for train_step in range(1, train_step + 1):
            LOSS = []
            # forward
            for idx in range(cli_nums):
                LOSS.append(local_nets[idx].train_forward(idx, train_step, train_datas[idx], train_step))
            # backward
            loss = awl(LOSS)
            loss.backward()
            # 更新参数
            for idx in range(cli_nums):
                local_nets[idx].optimizer.train_step()
                local_nets[idx].optimizer.zero_grad()
            awl_optimizer.train_step()
            awl_optimizer.zero_grad()

        # 一个epoch完成全部train_step,计算本轮的指标
        train_auc, train_loss = 0, 0
        for idx in range(cli_nums):
            # 获取指标
            local_nets[idx].train_metrics(idx, epoch, train_step, valid_datas[idx])
            train_auc += np.array(local_nets[idx].history['val_auc'])
            train_loss += np.array(local_nets[idx].history['val_loss'])
        # 联邦之前平均指标
        if lossnames == 'auc':
            print("before fed avg loss:", train_loss / cli_nums, "avg auc", train_auc / cli_nums)
        else:
            print("\n ERROR LOSS")

        if fed_type == 'FedDomainCross':
            # average_layers(local_nets)
            update_domain_emb_avg(local_nets, use_cuda, cross_type)
            # update_userEmb_avg(local_nets, client_data, uv_cli_info, fea_val_label, modletype, use_cuda,
            #                     same_dim)
        else:
            import warnings
            warnings.warn('fed argument wrong', UserWarning)

        # 联邦之后验证集平均指标
        fed_auc = pd.DataFrame()
        print("\nafter fed")
        for idx in range(cli_nums):
            # 客户端评估指标
            met = pd.DataFrame([local_nets[idx].evaluate(valid_datas[idx])])  # 返回的是字典
            fed_auc = pd.concat([fed_auc, met], axis=0)
            print("cli %d val_loss: %.4f val_auc: %.4f" % (idx + 1, met["val_loss"], met['val_auc']))

        if lossnames == 'auc':
            fed_auc = fed_auc.mean(axis=0)
            print("after fed avg loss: %.4f auc: %.4f" % (fed_auc["val_loss"], fed_auc['val_auc']))

        else:
            print("error loss")
    return 0, 1, 2, 3, 4


def fed_torchkeras2(Epochs, train_step, valid_step, fed_type, train_datas, valid_datas, test_datas, total_valid_datas, total_test_datas,
                    fea_embInfo,
                    uv_cli_info, client_data, fea_val_label, use_cuda, modletype, same_dim, test_type, lossnames,
                    cross_type):
    cli_nums = len(train_datas)
    local_nets = init_net(cli_nums, uv_cli_info, use_cuda, same_dim, fea_embInfo, modletype, lossnames, cross_type)
    net = local_nets[0]

    lr = 0.001
    awl = AutomaticWeightLoss(cli_nums)
    optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': lr, 'weight_decay':0.002},
        {'params': awl.parameters(), 'lr': lr}])
    net.compile(loss_func=net.net.lossfunc, optimizer=optimizer, metrics_dict=net.net.metricfunc)

    for epoch in range(1, Epochs+1):
        cli_train_metrics_sum = []
        cli_valid_metrics_sum = []
        for idx in range(cli_nums):
            train_metric = {'loss': 0.0}
            valid_metric = {'val_loss': 0.0}
            for name in net.metrics_dict.keys():
                train_metric[name] = 0.0
                name = "val_"+name
                valid_metric[name] = 0.0
            cli_train_metrics_sum.append(train_metric)
            cli_valid_metrics_sum.append(valid_metric)

        # train with train_step
        for step in range(1, train_step + 1):
            features, labels = net.get_train_step_data(step, train_datas)
            train_metrics = net.my_train_step(features, labels, awl)
            for idx, metric in enumerate(train_metrics):
                for name, val in metric.items():
                    cli_train_metrics_sum[idx][name] = cli_train_metrics_sum[idx].get(name, 0.0) + val
            if step % 35 == 0:
                for idx, cli_met_sum in enumerate(cli_train_metrics_sum,1):
                    logs = {"epoch": epoch, "cli": idx, "train_step": step}
                    logs.update({k: round(v / step, 4) for k, v in cli_met_sum.items()})
                    print(logs)
        # valid with train_step
        for step in range(1, valid_step+1):
            features, labels = net.get_train_step_data(step, valid_datas)
            valid_metrics = net.my_valid_train_step(features, labels)
            for idx, metric in enumerate(valid_metrics):
                for name, val in metric.items():
                    cli_valid_metrics_sum[idx][name] = cli_valid_metrics_sum[idx].get(name, 0.0) + val
        for idx, cli_met_sum in enumerate(cli_valid_metrics_sum,1):
            log = {"valid cli": idx}
            log.update({k: round(v / valid_step, 4) for k, v in cli_met_sum.items()})
            print(log)
        print('\n')


    return 0, 1, 2, 3, 4


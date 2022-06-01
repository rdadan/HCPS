import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

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


def update_user_emb_avg(nets, clientsdata, uv_cli_infos, fea_val_label, modletype, use_cuda, same_dim):
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
                    emb_name = 'net.embed_user_id.weight'
                avg_emb, fea_labels = get_avg_emb(nets, clis, fea, fea_val, emb_name, fea_val_label, use_cuda, same_dim)
                # 更新
                for cli, fea_label in zip(clis, fea_labels):
                    nets[cli].state_dict()[emb_name][fea_label].copy_(avg_emb)
            # 只聚合user_id
            if modletype == 'mlp' or modletype == 'mynet':
                break


def update_item_emb_avg(nets, clientsdata, uv_cli_infos, fea_val_label, modletype, use_cuda, same_dim):
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


def get_fea_emb_info_same(data_all, features):
    fed_emb_infos = {}
    for fea in features:
        fed_emb_infos[fea] = data_all[fea].nunique()
    return fed_emb_infos


def get_fea_emb_info_diff(cli):
    features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']
    # 每个客户端的emb div,以及特征值lable编码
    fea_val_lable_dict = {}
    # fea_emb_map = {'genre': 216}  # key:valus
    fea_emb_dim = {}
    for fea in features:
        # 每个特征的value值
        fea_value = cli[fea].unique()
        # 真实值转为label值
        le = LabelEncoder()
        cli[fea] = le.fit_transform(cli[fea])
        fea_label = cli[fea].unique()
        fea_val_lable_dict[fea] = dict(zip(fea_value, fea_label))
        fea_emb_dim[fea] = cli[fea].nunique()
    return cli, fea_val_lable_dict, fea_emb_dim


def create_dataloader(df, lossname, modle, use_cuda):
    # 4以上转换为positive，其他转换为negative
    global tensor_x, tensor_y
    df_tensor = torch.tensor(df.values, dtype=torch.float32)
    if use_cuda:
        df_tensor = df_tensor.cuda()
    # if modle == 'pnn':
    #     tensor_x, tensor_y = torch.split(df_tensor, [7, 1], dim=1)
    if modle in ['mlp', 'pnn', 'dcn', 'metamf', 'mynet']:
        tensor_x, tensor_y = torch.split(df_tensor, [2, 1], dim=1)
    else:
        import warnings
        warnings.warn('fed modle wrong', UserWarning)

    # 根据Tensor创建数据集
    dataset = TensorDataset(tensor_x, tensor_y.view(-1))
    # 使用DataLoader加载数据集
    train_data = DataLoader(dataset, shuffle=True, batch_size=128, drop_last=True)
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


def readdata_ML100k(path,train_step, valid_step, lossnames, modle, use_cuda, is_same_emb=True, istest=True, file=None):
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
    if modle in ['mlp', 'pnn', 'dcn', 'metamf', 'mynet']:
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
    fea_emb_info = []
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
            df, fea_val_lable_dict, fea_emb_dim = get_fea_emb_info_diff(df)
            fea_value_label.append(fea_val_lable_dict)
            fea_emb_info.append(fea_emb_dim)
        # 分成训练集，验证集，测试集

        train_df = df.sample(frac=0.8, random_state=123, replace=False)
        other = df[~df.index.isin(train_df.index)]
        valid_df = other.sample(frac=0.5, random_state=123, replace=False)
        total_valid_data = total_valid_data.append(valid_df, ignore_index=True)
        test_df = other[~other.index.isin(valid_df.index)]
        total_test_data = total_test_data.append(test_df, ignore_index=True)
        # DataLoader加载
        train = create_dataloader(train_df, lossname, modle, use_cuda)
        train_datas.append(train)
        valid = create_dataloader(valid_df, lossname, modle, use_cuda)
        valid_datas.append(valid)
        test = create_dataloader(test_df, lossname, modle, use_cuda)
        test_datas.append(test)
    print("df total_len: ", total_len)
    total_valid_datas = create_dataloader(total_valid_data, lossname, modle, use_cuda)
    total_test_datas = create_dataloader(total_test_data, lossname, modle, use_cuda)
    if is_same_emb:
        fea_emb_info = get_fea_emb_info_same(all_data, cols[:-1])
    # uv_cli_info = read_uv_cli_info(path)
    uv_cli_info = get_uv_cli_info(all_data, client_data)
    return train_datas, valid_datas, test_datas, total_valid_datas, total_test_datas, fea_emb_info, uv_cli_info, client_data, fea_value_label


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

import torch.nn as nn
import os

os.environ['CUDA_ENABLE_DEVICES'] = '0'


# 定义一个全连接层的神经网络
class DNN(nn.Module):
    def __init__(self, hidden_units, dropout=0.3):
        super(DNN, self).__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)

    # 前向传播， 遍历dnn_network， 加激活函数
    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, emb_info, hidden_units, lossname):
        super(MLP, self).__init__()
        self.embed_user_id = nn.Embedding(num_embeddings=emb_info['user_id'], embedding_dim=10)
        self.embed_movie_id = nn.Embedding(num_embeddings=emb_info['movie_id'], embedding_dim=10)
        """ fully connected layer """
        # self.MLP_Layers = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(Layers[:-1], Layers[1:]))])
        self.dnn_network = DNN(hidden_units)
        self.dense_final = nn.Linear(hidden_units[-1], 1)
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
        embed_user_id = self.embed_user_id(inputs[:, 0])
        embed_movie_id = self.embed_movie_id(inputs[:, 1])
        """ 拼接 """
        embedding_cat = torch.cat((embed_user_id, embed_movie_id), dim=1)
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
    def __init__(self, user_num, item_num, item_emb_size=32, item_mem_num=8, user_emb_size=32, domain_mem=64,
                 hidden_size=64):  # note that we have many users and each user has many layers
        super(MetaRecommender, self).__init__()
        self.item_num = item_num
        self.item_emb_size = item_emb_size
        self.item_mem_num = item_mem_num
        # For each user
        self.user_embedding = nn.Embedding(user_num, user_emb_size)
        self.memory = torch.nn.Parameter(nn.init.xavier_normal_(torch.Tensor(user_emb_size, domain_mem)),
                                         requires_grad=True)
        # For each layer
        self.hidden_layer_1, self.weight_layer_1, self.bias_layer_1 = self.define_one_layer(domain_mem, hidden_size,
                                                                                            item_emb_size,
                                                                                            int(item_emb_size / 4))
        self.hidden_layer_2, self.weight_layer_2, self.bias_layer_2 = self.define_one_layer(domain_mem, hidden_size,
                                                                                            int(item_emb_size / 4), 1)
        self.hidden_layer_3, self.emb_layer_1, self.emb_layer_2 = self.define_item_embedding(item_num, item_emb_size,
                                                                                             item_mem_num, domain_mem,
                                                                                             hidden_size)

    def define_one_layer(self, domain_mem, hidden_size, int_size, out_size):  # define one layer in MetaMF
        hidden_layer = nn.Linear(domain_mem, hidden_size)
        weight_layer = nn.Linear(hidden_size, int_size * out_size)
        bias_layer = nn.Linear(hidden_size, out_size)
        return hidden_layer, weight_layer, bias_layer

    def define_item_embedding(self, item_num, item_emb_size, item_mem_num, domain_mem, hidden_size):
        hidden_layer = nn.Linear(domain_mem, hidden_size)
        emb_layer_1 = nn.Linear(hidden_size, item_num * item_mem_num)
        emb_layer_2 = nn.Linear(hidden_size, item_mem_num * item_emb_size)
        return hidden_layer, emb_layer_1, emb_layer_2

    def forward(self, user_id):
        # collaborative memory module
        user_emb = self.user_embedding(user_id)  # input_user=[batch_size, user_emb_size]
        cf_vec = torch.matmul(user_emb,
                              self.memory)  # [user_num, u_emb_size]*[u_emb_size, domain_mem]=[batch_size, domain_mem]
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
        return output_weight, output_bias, item_embedding, cf_vec  # ([len(layer_list)+1, batch_size, *, *], [len(layer_list)+1, batch_size, 1, *], [batch_size, item_num, item_emb_size], [batch_size, domain_mem])

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
    def __init__(self, emb_info, lossname, item_emb_size=10, item_mem_num=8, user_emb_size=10, domain_mem=64,
                 hidden_size=64):
        super(MetaMF, self).__init__()
        self.item_num = emb_info['movie_id']
        self.metarecommender = MetaRecommender(emb_info['user_id'], emb_info['movie_id'], item_emb_size,
                                               item_mem_num, user_emb_size, domain_mem, hidden_size)
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
        model_weight, model_bias, item_embedding, _ = self.metarecommender(inputs[:, 0])
        item_id = inputs[:, 1]
        item_id = item_id.view(-1, 1)  # item_id=[batch_size, 1]
        item_one_hot = torch.zeros(len(item_id), self.item_num,
                                   device=item_id.device)  # we generate it dynamically, and default device is cpu
        item_one_hot.scatter_(1, item_id, 1)  # item_one_hot=[batch_size, item_num]
        item_one_hot = torch.unsqueeze(item_one_hot, 1)  # item_one_hot=[batch_size, 1, item_num]
        item_emb = torch.matmul(item_one_hot, item_embedding)  # out=[batch_size, 1, item_emb_size]
        out = torch.matmul(item_emb, model_weight[0])  # out=[batch_size, 1, item_emb_size/4]
        out = out + model_bias[0]  # out=[batch_size, 1, item_emb_size/4]
        out = F.relu(out)  # out=[batch_size, 1, item_emb_size/4]
        out = torch.matmul(out, model_weight[1])  # out=[batch_size, 1, 1]
        out = out + model_bias[1]  # out=[batch_size, 1, 1]
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
    def __init__(self, emb_info, hidden_units, layer_num, use_cuda, lossname, emb_dim=10):
        super(DCN, self).__init__()
        # self.dense_emb_info, self.emb_info = emb_info
        self.emb_info = emb_info
        # embedding层，类别特征embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=emb_dim)
            for key, val in emb_info.items()
        })

        hidden_units.insert(0, len(emb_info) * emb_dim)
        self.cross_network = CrossDcn(layer_num, hidden_units[0])  # layer_num是交叉网络的层数， hidden_units[0]表示输入的整体维度大小
        self.dnn_network = DNN(hidden_units)
        self.dense_final = nn.Linear(hidden_units[-1] + hidden_units[0], 1)
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
                         for key, i in zip(self.emb_info.keys(), range(inputs.shape[1]))]
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

    def __init__(self, emb_dim, sparse_num, hidden_units, use_cuda):
        super(CrossPnn, self).__init__()
        self.w_z = nn.Parameter(torch.rand([sparse_num, emb_dim, hidden_units[0]]))
        # p部分
        self.w_p = nn.Parameter(torch.rand([sparse_num, sparse_num, hidden_units[0]]))  # [26,26,256]
        self.l_b = torch.rand([hidden_units[0], ], requires_grad=True)
        if use_cuda:
            self.l_b = self.l_b.cuda()

    def forward(self, z, inputs_embeds):
        # l_z = torch.mm([128,70], [70,64])=[128,64]
        l_z = torch.mm(z.reshape(z.shape[0], -1),
                       self.w_z.permute((2, 0, 1)).reshape(self.w_z.shape[2], -1).T)  # (None, hidden_units[0])
        # matmul可处理维度不同的矩阵, : [2,5,3]*[1,3,4]->[2,5,4]
        # lp = matmul([128,7,10], [128, 10, 7])=[128, 7, 7]
        p = torch.matmul(inputs_embeds, inputs_embeds.permute((0, 2, 1)))  # [None, sparse_num, sparse_num]
        # mm([128,49],[49,64])=[128,64]
        l_p = torch.mm(p.reshape(p.shape[0], -1),
                       self.w_p.permute((2, 0, 1)).reshape(self.w_p.shape[2], -1).T)  # [None, hidden_units[0]]
        # output = [128,64]+[128,64]+[64]=[128,64]
        output = l_z + l_p + self.l_b
        return output


# PNN网络
# 逻辑是底层输入（类别型特征) -> embedding层 -> crosspnn 层 -> DNN -> 输出
class PNN(nn.Module):
    # hidden_units = [256, 128, 64]
    def __init__(self, emb_info, hidden_units, use_cuda, lossname, mode_type='in', dnn_dropout=0.3, emb_dim=10):
        super(PNN, self).__init__()
        self.features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']
        self.emb_info = emb_info
        # self.dense_num = len(self.dense_feas)  # 13 L
        self.sparse_num = len(emb_info)  # 26 C
        self.mode_type = mode_type
        self.emb_dim = emb_dim

        # embedding层，类别特征embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=self.emb_dim)
            for key, val in self.emb_info.items()
        })

        # crosspnn层
        self.crosspnn = CrossPnn(emb_dim, self.sparse_num, hidden_units, use_cuda)

        # dnn 层
        # hidden_units[0] += self.dense_num  # dense_inputs直接输入道dnn，没有embedding 256+13=269
        self.dnn_network = DNN(hidden_units, dnn_dropout)
        self.dense_final = nn.Linear(hidden_units[-1], 1)
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
        #                  for key, i in zip(self.emb_info.keys(), range(inputs.shape[1]))]
        user_emb = self.embed_layers['embed_user_id'](inputs[:, 0])
        inputs_embeds.append(user_emb)
        movie_emb = self.embed_layers['embed_movie_id'](inputs[:, 1])
        inputs_embeds.append(movie_emb)
        inputs_embeds = torch.stack(inputs_embeds)  # [fea_num, batch_sz, emb_dim]->[7,128,10]
        # [None, sparse_num, emb_dim]  此时空间不连续， 下面改变形状不用view，用reshape
        inputs_embeds = inputs_embeds.permute((1, 0, 2))  # [batch_sz, fea_num, emb_dim]->[128,7,10]
        # crosspnn layer foward
        cross_outputs = self.crosspnn(inputs_embeds, inputs_embeds)
        l1 = F.relu(cross_outputs)
        # dnn_network
        dnn_x = self.dnn_network(l1)
        outputs = self.dense_final(dnn_x)
        outputs = self.actifunc(outputs)
        outputs = outputs.view(-1)
        return outputs


def get_uv_embedding(emb_layer_1, emb_layer_2, cf_vec, domain_num, domain_mem_num, domain_emb_size, droprate):
    emb_left = emb_layer_1(cf_vec)
    emb_right = emb_layer_2(cf_vec)
    emb_left = emb_left.view(-1, domain_num, domain_mem_num)  # emb_left=[batch_size, domain_num, domain_mem_num]
    emb_right = emb_right.view(-1, domain_mem_num,
                               domain_emb_size)  # emb_right=[batch_size, domain_mem_num, domain_emb_size]
    private_domain_emb = torch.matmul(emb_left, emb_right)
    return private_domain_emb


def set_uv_emb_layer(domain_num, emb_dim, mem_num, domain_mem):
    emb_layer_1 = nn.Linear(domain_mem, domain_num * mem_num)
    emb_layer_2 = nn.Linear(domain_mem, mem_num * emb_dim)
    return emb_layer_1, emb_layer_2


class DomainLayer(nn.Module):  # in fact, it's not a hypernetwork
    def __init__(self, domain_num, emb_dim, mem_num, mem_size, droprate):
        super(DomainLayer, self).__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.mem_num = mem_num
        self.droprate = droprate
        self.domain_embedding = nn.Embedding(domain_num, emb_dim)
        self.shared_memory_d = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(emb_dim, mem_size)), requires_grad=True)
        self.emb_l1, self.emb_l2 = set_uv_emb_layer(domain_num, emb_dim, mem_num, mem_size)

    def forward(self, domain_id):
        domain_emb = self.domain_embedding(domain_id)
        domain_cf_vec_d = torch.matmul(domain_emb, self.shared_memory_d)  # [1,emb_dim]*[emb_dim,mem_size]=[1,mem_size]
        d_embs = get_uv_embedding(self.emb_l1, self.emb_l2, domain_cf_vec_d,
                                  self.domain_num, self.mem_num, self.emb_dim, self.droprate)  # 【1，18，10】
        return d_embs.squeeze(0)


class myNet(nn.Module):
    # hidden_units = [256, 128, 64]
    def __init__(self, emb_info, domain_id, uv_domains_info, hidden_units, use_cuda, lossname, cross_type,
                 dropout=0.0, emb_dim=32, domain_mem=512, mem_size=32):
        super(myNet, self).__init__()
        # self.features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']
        hidden_units = [64, 64, 32, 16]

        self.use_cuda = use_cuda
        self.emb_info = emb_info
        self.cross_type = cross_type
        self.domain_id = domain_id
        self.uv_domains_info = uv_domains_info
        self.embed_user_id = nn.Embedding(num_embeddings=emb_info['user_id'], embedding_dim=emb_dim)
        self.embed_movie_id = nn.Embedding(num_embeddings=emb_info['movie_id'], embedding_dim=emb_dim)

        if cross_type == 'crossmy':
            # emb_dim = emb_dim * 2
            self.private_domain = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(1, domain_mem)))
            self.shared_memory_d = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(domain_mem, emb_dim)))
            self.domain_weights_u = nn.Parameter(torch.rand(emb_dim, 1))
            self.domain_bias_u = nn.Parameter(torch.rand(emb_dim, 1))
            self.domain_weights_v = nn.Parameter(torch.rand(emb_dim, 1))
            self.domain_bias_v = nn.Parameter(torch.rand(emb_dim, 1))
            # self.losswi = nn.Parameter(torch.rand(1, 1))
        elif cross_type == 'crosspnn':
            self.domain_embedding = nn.Embedding(emb_info['domain_id'], emb_dim)
            self.domain = DomainLayer(emb_info['domain_id'], emb_dim, mem_size, domain_mem, dropout)
            self.mycross = CrossPnn(emb_dim, 4, hidden_units, use_cuda)  # sparse_num=4
        elif cross_type == 'crossdcn':
            self.domain_embedding = nn.Embedding(emb_info['domain_id'], emb_dim)
            self.domain = DomainLayer(emb_info['domain_id'], emb_dim, mem_size, domain_mem, dropout)
            self.mycross = CrossDcn(3, hidden_units[0])  # cross_layer_num=3
        # dnn 层
        self.dnn_network = DNN(hidden_units, dropout)
        self.dense_final = nn.Linear(hidden_units[-1], 1)
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
        user_emb = self.embed_user_id(inputs[:, 0])
        item_emb = self.embed_movie_id(inputs[:, 1])
        # 域
        if self.cross_type == 'crossmy':
            share_domain = torch.matmul(self.private_domain, self.shared_memory_d)
            emb_u = torch.unsqueeze(user_emb, dim=2)
            emb_v = torch.unsqueeze(item_emb, dim=2)
            emb_u = torch.matmul(torch.matmul(emb_u, share_domain), self.domain_weights_u) + self.domain_bias_u + emb_u
            emb_v = torch.matmul(torch.matmul(emb_v, share_domain), self.domain_weights_v) + self.domain_bias_v + emb_v
            emb = torch.cat((emb_u, emb_v), 1)
            ## emb = F.relu(emb)
            # emb = torch.cat((user_emb, item_emb), 1)
            l1 = torch.squeeze(emb)
        elif self.cross_type == 'crosscat':
            l1 = torch.cat((user_emb, item_emb), 1)
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
        l1 = self.dnn_network(l1)
        outputs = self.dense_final(l1)
        outputs = self.actifunc(outputs)
        outputs = outputs.view(-1)
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

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy
from sklearn.metrics import roc_auc_score
import datetime


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


# torchkeras
def init_net(cli_nums, uv_cli_info, use_cuda, same_dim, fea_emb_info, modletype, lossnames, cross_type):
    global net
    import torchkeras
    hidden_units = [64, 32, 16]
    if cross_type == 'torchcat': hidden_units = [20, 64, 32, 16]
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
        emb_info = fea_emb_info if same_dim else fea_emb_info[idx]
        if modletype == 'mlp':
            hidden_units = [20, 64, 32, 16]
            net = MLP(emb_info, hidden_units, lossname)
        elif modletype == 'metamf':
            net = MetaMF(emb_info, lossname)
        elif modletype == 'pnn':
            net = PNN(emb_info, hidden_units, use_cuda, lossname)
        elif modletype == 'dcn':
            net = DCN(emb_info, hidden_units, 3, use_cuda, lossname)
        elif modletype == 'mynet':
            domain_id = torch.tensor(idx).long()
            if use_cuda: domain_id = domain_id.cuda()
            emb_info['domain_id'] = cli_nums
            hidden_units = [20, 64, 32, 16]

            net = myNet(emb_info, domain_id, uv_cli_info, hidden_units, use_cuda, lossname, cross_type)
        else:
            import warnings
            warnings.warn('modletype argument wrong', UserWarning)

        if use_cuda: net.cuda()
        net.apply(weights_init)
        net = torchkeras.Model(net)
        local_nets.append(net)
    return local_nets


def fed_torchkeras(T, train_step, valid_step, fed_type, train_datas, valid_datas, test_datas, total_valid_datas, total_test_datas, fea_emb_info,
                   uv_cli_info, client_data, fea_val_label, use_cuda, modletype, same_dim, test_type, lossnames,
                   cross_type):
    cli_nums = len(train_datas)
    local_nets = init_net(cli_nums, uv_cli_info, use_cuda, same_dim, fea_emb_info, modletype, lossnames, cross_type)
    # 全部客户端的metrics记录格式
    # cols = ["epoch", "loss", "auc", "val_loss", "val_auc", "fed_loss", "fed_auc"]
    #
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

    df_cli = pd.DataFrame(columns=cols_cli)
    df_auc = pd.DataFrame(columns=cols_auc)
    df_mae = pd.DataFrame(columns=cols_mae)

    # 联邦训练
    lr = 0.001
    for t in range(T):
        train_auc = pd.DataFrame()
        train_mae = pd.DataFrame()

        if (t + 1) % 3 == 0:
            lr = float("%.4f" % (lr * np.power(0.94, 4)))
            # lr = round(lr * np.power(0.97, (T / 3)), 4)
        print("decay_lr: ", lr)
        for idx in range(cli_nums):
            print(
                "------------------------------------------------------------------------------------------------client",
                idx + 1, modletype, fed_type, cross_type, "t", t + 1)
            # 参数 torch.nn.BCELoss()
            local_nets[idx].compile(loss_func=local_nets[idx].net.lossfunc,
                                    optimizer=torch.optim.Adam(params=local_nets[idx].net.parameters(), lr=lr,
                                                               weight_decay=0.001),
                                    metrics_dict=local_nets[idx].net.metricfunc)
            # 训练
            epo = 1
            met = local_nets[idx].fit(epochs=epo, dl_train=train_datas[idx], dl_val=valid_datas[idx],
                                      log_step_freq=3000)
            # 拼接 epoch | loss |  mae  | mse | rmse |  val_mae  | val_mse | val_rmse
            # df_cli.loc[t, strcol] = "%.4f" % met['mae']
            strcol = "train_cli_" + str(idx)
            met = pd.DataFrame(met.iloc[epo - 1, :]).T
            # val_met = local_nets[idx].net.lossname
            # df_cli.loc[t, strcol] = float("%.4f" % met[val_met])
            val = met.iloc[0][3]
            df_cli.loc[t, strcol] = float("%.4f" % val)
            lossname = local_nets[idx].net.lossname
            if lossname == 'auc':
                train_auc = pd.concat([train_auc, met], axis=0)
            elif lossname == 'mae':
                train_mae = pd.concat([train_mae, met], axis=0)
            # 保存模型
            if fed_type in ['Central', 'FedDomainCat'] and (t + 1) % 1 == 0 and float("%.4f" % val) < best_net[idx][0]:
                print("Central save net")
                best_net[idx][0] = float("%.4f" % val)
                best_net[idx][1] = local_nets[idx]

        # 联邦之前各客户端平均指标
        # ['loss', 'auc', 'val_loss', 'val_auc', 'mae', 'val_mae']
        # mtrain_avg = mtrain_avg.mean(axis=0)
        if lossnames == 'auc':
            train_auc = train_auc.mean(axis=0)
            print("\navg auc before fed: \nloss: %.4f valid loss: %.4f auc: %.4f val auc: %.4f" % (
                train_auc['loss'], train_auc['val_loss'], train_auc['auc'], train_auc['val_auc']))
        elif lossnames == 'mae':
            train_mae = train_mae.mean(axis=0)
            print("\navg mae before fed: \nloss: %.4f valid loss: %.4f mae: %.4f val mae: %.4f" % (
                train_mae['loss'], train_mae['val_loss'], train_mae['mae'], train_mae['val_mae']))
        elif lossnames == 'none':
            train_auc = train_auc.mean(axis=0)
            train_mae = train_mae.mean(axis=0)

            print("\navg auc before fed: \nloss: %.4f valid loss: %.4f auc: %.4f val auc: %.4f" % (
                train_auc['loss'], train_auc['val_loss'], train_auc['auc'], train_auc['val_auc']))
            print("\navg mae before fed: \nloss: %.4f valid loss: %.4f mae: %.4f val mae: %.4f" % (
                train_mae['loss'], train_mae['val_loss'], train_mae['mae'], train_mae['val_mae']))
        else:
            print("\n ERROR LOSS")

        # 联邦聚合
        if fed_type == "FedEmb":
            update_user_emb_avg(local_nets, client_data, uv_cli_info, fea_val_label, modletype, use_cuda, same_dim)
            update_item_emb_avg(local_nets, client_data, uv_cli_info, fea_val_label, modletype, use_cuda, same_dim)
            # local_nets = average_other_layers(local_nets)
        elif fed_type == "FedAvg":
            average_all_layers(local_nets)
        elif fed_type == 'FedDomainCross':
            # average_layers(local_nets)
            update_domain_emb_avg(local_nets, use_cuda, cross_type)
            # update_user_emb_avg(local_nets, client_data, uv_cli_info, fea_val_label, modletype, use_cuda,
            #                     same_dim)
        elif fed_type == 'FedDomainCat':
            # df_avg.loc[t] = (t + 1, mtrain_avg["loss"], mtrain_avg[tra_met], mtrain_avg["val_loss"], mtrain_avg[val_met])
            continue
        elif fed_type == 'Central':
            # df_avg.loc[t] = (t + 1, mtrain_avg["loss"], mtrain_avg[tra_met], mtrain_avg["val_loss"], mtrain_avg[val_met])
            continue
        else:
            import warnings
            warnings.warn('fed argument wrong', UserWarning)
        # 联邦之后反向传播
        print("\nafter fed")
        for idx in range(cli_nums):
            # backward
            local_nets[idx].net.train()
            #loss = local_nets[idx].net.lossfunc
            # loss.item = train_auc['loss']
            optimizer = torch.optim.Adam(params=local_nets[idx].net.parameters(), lr=lr,
                                         weight_decay=0.001)
            loss = torch.tensor(train_auc['loss'], requires_grad=True)
            loss.backward()
            # update parameters
            optimizer.step()
            optimizer.zero_grad()
            print("\nafter fed")
        # 联邦之后反向传播，客户端评估指标
        fed_auc = pd.DataFrame()
        fed_mae = pd.DataFrame()
        for idx in range(cli_nums):
            met = pd.DataFrame([local_nets[idx].evaluate(valid_datas[idx])])  # 返回的是字典
            val_ = met.iloc[0][1]
            val_ = float("%.4f" % val_)
            df_cli.loc[t]["fed_cli_" + str(idx)] = val_
            lossname = local_nets[idx].net.lossname
            print("cli:%d loss: %5f  %s:%5f" % (idx + 1, met["val_loss"], lossname, val_))
            # 保存模型
            if t == 0:
                best_net[idx][0] = val_
            if lossname == 'auc':
                if (t + 1) % 2 == 0 and val_ > best_net[idx][0]:
                    best_net[idx][0] = val_
                    best_net[idx][1] = local_nets[idx]
                fed_auc = pd.concat([fed_auc, met], axis=0)
            elif lossname == 'mae':
                fed_mae = pd.concat([fed_mae, met], axis=0)
                if (t + 1) % 2 == 0 and val_ < best_net[idx][0]:
                    best_net[idx][0] = val_
                    best_net[idx][1] = local_nets[idx]

        # 计算验证集的平均指标
        if lossnames == 'none':
            fed_auc = fed_auc.mean(axis=0)
            fed_mae = fed_mae.mean(axis=0)
            print("fed auc avg loss: %.4f auc: %.4f" % (fed_auc["val_loss"], fed_auc['val_auc']))
            print("fed mae avg loss: %.4f mae: %.4f" % (fed_mae["val_loss"], fed_mae['val_mae']))

            df_auc.loc[t] = (t + 1, train_auc["loss"], train_auc['auc'],
                             train_auc["val_loss"], train_auc['val_auc'],
                             fed_auc["val_loss"], fed_auc['val_auc'])

            df_mae.loc[t] = (t + 1, train_mae["loss"], train_mae['mae'],
                             train_mae["val_loss"], train_mae['val_mae'],
                             fed_mae["val_loss"], fed_mae['val_mae'])
        elif lossnames == 'mae':
            fed_mae = fed_mae.mean(axis=0)
            print("fed mae avg loss: %.4f mae: %.4f" % (fed_mae["val_loss"], fed_mae['val_mae']))
            df_mae.loc[t] = (t + 1, train_mae["loss"], train_mae['mae'],
                             train_mae["val_loss"], train_mae['val_mae'],
                             fed_mae["val_loss"], fed_mae['val_mae'])
        elif lossnames == 'auc':
            fed_auc = fed_auc.mean(axis=0)
            print("fed auc avg loss: %.4f auc: %.4f" % (fed_auc["val_loss"], fed_auc['val_auc']))
            df_auc.loc[t] = (t + 1, train_auc["loss"], train_auc['auc'],
                             train_auc["val_loss"], train_auc['val_auc'],
                             fed_auc["val_loss"], fed_auc['val_auc'])
        else:
            print("error loss")



    print("\ntest model:")
    # 测试模型
    test_auc = pd.DataFrame()
    test_mae = pd.DataFrame()
    for idx in range(cli_nums):
        # 客户端评估指标
        met = pd.DataFrame([best_net[idx][1].evaluate(test_datas[idx])])  # 返回的是字典
        lossname = best_net[idx][1].net.lossname
        if lossname == 'auc':
            test_auc = pd.concat([test_auc, met], axis=0)
        elif lossname == 'mae':
            test_mae = pd.concat([test_mae, met], axis=0)
        print("cli:%d loss: %5f  %s:%5f" % (idx + 1, met["val_loss"], lossname, met.iloc[0][1]))
    # 计算测试集上平均指标

    if lossnames == 'none':
        test_auc = test_auc.mean(axis=0)
        test_mae = test_mae.mean(axis=0)
        print("auc avg loss: %.4f auc: %.4f" % (test_auc["val_loss"], test_auc['val_auc']))
        print("mae avg loss: %.4f mae: %.4f" % (test_mae["val_loss"], test_mae['val_mae']))
    elif lossnames == 'mae':
        test_mae = test_mae.mean(axis=0)
        print("mae avg loss: %.4f mae: %.4f" % (test_mae["val_loss"], test_mae['val_mae']))
    elif lossnames == 'auc':
        test_auc = test_auc.mean(axis=0)
        print("auc avg loss: %.4f auc: %.4f" % (test_auc["val_loss"], test_auc['val_auc']))
    else:
        print("error loss")
    return best_net, df_auc, df_mae, df_cli, cols_cli

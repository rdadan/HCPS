import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

"=================================================================================================================================================ulitiy"


def plot_metric2(dfhistory, figname, is_test=False):
    # epoch      loss       auc  val_loss   val_auc  fed_loss   fed_auc
    plt.title(figname)
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    # cols = ["epoch",
    #         "loss", "mae", "mse", "rmse",
    #         "val_loss", "val_mae", "val_mse", "val_rmse",
    #         "fed_loss", "fed_mae", "fed_mse", "fed_rmse"]
    cols = ['mae', 'val_mae', 'fed_mae']
    dfhistory = dfhistory[cols]
    # plt.plot(dfhistory.values[:, 5:], label=dfhistory.columns.values[1:])
    plt.plot(dfhistory, label=cols)

    plt.legend()
    if is_test is False:
        picture = figname + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
        plt.savefig(picture, dpi=800)

    max_val_mae = dfhistory["val_mae"].min()
    print("min_val_mae", max_val_mae)
    max_fed_auc = dfhistory["fed_mae"].min()
    print("min_fed_mae", max_fed_auc)

    plt.show()


def average_layers(nets):
    state_dict = [x.state_dict() for x in nets]
    keys = list(state_dict[0].keys())
    for k in keys:
        if k.__contains__("emb") is True:
            weights = 0
            for i in range(len(nets)):
                weights = weights + state_dict[i][k]
            for net in nets:
                net.state_dict()[k].copy_(weights / len(nets))


def average_other_layers(nets):
    state_dict = [x.state_dict() for x in nets]
    keys = list(state_dict[0].keys())
    for k in keys:
        if k.__contains__("emb") is False:  # or k.__contains__("genre") is True:
            weights = 0
            for i in range(len(nets)):
                weights = weights + state_dict[i][k]
            for net in nets:
                net.state_dict()[k].copy_(weights / len(nets))
    return nets


def updata_uv_emb(nets, cli_labels_dic, feal_label_dic, embs_dic):
    # 更新拥有该用户的cli的user_embedding
    for cli, feal_label_dic in cli_labels_dic.items():
        fea_idx = 0
        for fea, label in feal_label_dic.items():
            fea_name = "embed_layers.embed_" + fea + ".weight"
            nets[cli].state_dict()[fea_name][label].copy_(embs_dic[cli][fea_idx])
            fea_idx += 1


def update_item_emb(nets, clientsdata, uv_cli_infos, fea_label_dicts):
    # 更新item特征
    v_features = ['movie_id', 'genre']  # 'genre'是mulithot
    for vid, clis in uv_cli_infos[1].items():
        if len(clis) > 1:
            cli_labels_dic = {}
            embs_dic = {}
            genre_embs = []
            for i in clis:
                v_fea_values = clientsdata[i][clientsdata[i]["movie_id"] == vid].values[0][5:-1]
                feal_label_dic = {}
                fea_idx = 5
                for fea, fea_val in zip(v_features[:-1], v_fea_values):
                    feal_label_dic[fea] = fea_label_dicts[i][fea_idx][fea_val]
                    fea_idx += 1
                # genre的label在所有客户端中都是一样的
                feal_label_dic[v_features[-1]] = v_fea_values[-1]
                cli_labels_dic[i] = feal_label_dic
                embs = []
                for fea, label in feal_label_dic.items():
                    fea_name = "embed_layers.embed_" + fea + ".weight"
                    fea_emb = nets[i].state_dict()[fea_name][label].float()  # gpu
                    embs.append(fea_emb)
                embs_dic[i] = embs
                # # 取genre_mulit_hot
                # genre_mulit_hot = clientsdata[i][clientsdata[i]["movie_id"] == vid].values[0][-2]
                # genre_emb_name = "embed_layers.embed_genre.weight"
                # genre_emb = torch.matmul(genre_mulit_hot, nets[i].state_dict()[genre_emb_name])
                # genre_embs.append(genre_emb)

            # embs = torch.tensor([e.cpu().detach().numpy() for e in embs_dic.values()])  # .cpu().detach().numpy()
            embs = []
            for e in embs_dic.values():
                emb = torch.stack(e, 0)
                embs.append(emb)
            embs = torch.stack(embs, 0)
            avg_embs = torch.mean(embs, dim=0, keepdim=True).squeeze()
            for i in clis:
                embs_dic[i] = avg_embs

            # 更新拥有该item的cli的item相关embedding
            for cli, feal_label_dic in cli_labels_dic.items():
                fea_idx = 0
                for fea, label in feal_label_dic.items():
                    fea_name = "embed_layers.embed_" + fea + ".weight"
                    nets[cli].state_dict()[fea_name][label].copy_(embs_dic[cli][fea_idx])
                    fea_idx += 1
    return nets


def update_user_emb(nets, clientsdata, uv_cli_infos, fea_label_dicts, same_embs=True):
    u_features = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    for uid, clis in uv_cli_infos[0].items():
        if len(clis) > 1:
            cli_labels_dic = {}
            embs_dic = {}
            for i in clis:
                u_fea_values = clientsdata[i][clientsdata[i]["user_id"] == uid].values[0][5, 6]
                feal_label_dic = {}
                fea_idx = 0
                for fea, fea_val in zip(u_features, u_fea_values):
                    feal_label_dic[fea] = fea_label_dicts[i][fea_idx][fea_val]
                    fea_idx += 1
                cli_labels_dic[i] = feal_label_dic
                embs = []
                for fea, label in feal_label_dic.items():
                    fea_name = "embed_layers.embed_" + fea + ".weight"
                    fea_emb = nets[i].state_dict()[fea_name][label].float()  # gpu
                    embs.append(fea_emb)
                embs_dic[i] = embs

            embs = [torch.stack(e, 0) for e in embs_dic.values()]

            #     for e in embs_dic.values():
            #         embs.append(torch.stack(e, 0))
            embs = torch.stack(embs, 0)
            avg_embs = torch.mean(embs, dim=0, keepdim=True).squeeze()
            embs_dic = zip(clis, avg_embs)
            #             for i in clis:
            #                 embs_dic[i] = avg_embs
            nets = updata_uv_emb(nets, cli_labels_dic, feal_label_dic, embs_dic)

    return nets


def update_user_emb__diffdim_sum(nets, clientsdata, uv_cli_infos, fea_label_dicts):
    u_features = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    for uid, clis in uv_cli_infos[0].items():
        if len(clis) > 1:
            cli_labels_dic = {}
            embs_dic = {}
            records_num = {}
            for i in clis:
                user_data = clientsdata[i][clientsdata[i]["user_id"] == uid].values
                records_num[i] = (len(user_data))
                u_fea_values = user_data[0][:5]
                feal_label_dic = {}
                fea_idx = 0
                for fea, fea_val in zip(u_features, u_fea_values):
                    feal_label_dic[fea] = fea_label_dicts[i][fea_idx][fea_val]
                    fea_idx += 1
                cli_labels_dic[i] = feal_label_dic
                embs = []
                for fea, label in feal_label_dic.items():
                    fea_name = "embed_layers.embed_" + fea + ".weight"
                    fea_emb = nets[i].state_dict()[fea_name][label].float()  # gpu
                    embs.append(fea_emb)
                embs_dic[i] = embs
            embs = []
            for e in embs_dic.values():
                emb = torch.stack(e, 0)
                embs.append(emb)
            embs = torch.stack(embs, 0)
            # avg_embs = torch.mean(embs, dim=0, keepdim=True).squeeze()
            sum_embs = torch.sum(embs, dim=0, keepdim=True).squeeze()

            for i in clis:
                embs_dic[i] = sum_embs
            # 更新拥有该用户的cli的user_embedding
            for cli, feal_label_dic in cli_labels_dic.items():
                fea_idx = 0
                for fea, label in feal_label_dic.items():
                    fea_name = "embed_layers.embed_" + fea + ".weight"
                    nets[cli].state_dict()[fea_name][label].copy_(
                        (records_num[cli] / sum(records_num.values())) * embs_dic[cli][fea_idx])
                    fea_idx += 1
    return nets


def get_avg_emb(nets, clis, fea, fea_val, emb_name, fea_val_label, use_cuda, same_dim):
    embs = []
    w_p = []
    w_z = []
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
                avg_emb, fea_labels = get_avg_emb(nets, clis, fea, fea_val, emb_name, fea_val_label, use_cuda, same_dim)
                # 更新
                for cli, fea_label in zip(clis, fea_labels):
                    nets[cli].state_dict()[emb_name][fea_label].copy_(avg_emb)
                if modletype == 'mlp':
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
                if modletype == 'mlp':
                    break


"=================================================================================================================================================readdata"

import itertools
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder


def get_fea_emb_dims_same(data_all):
    features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']
    fed_emb_infos = {}
    for fea in features:
        fed_emb_infos[fea] = data_all[fea].nunique()
    return fed_emb_infos


def get_fea_emb_dims_diff(cli):
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
    # fea_value_label.append(fea_dict)
    # fea_emb_dim.append(fea_emb_map)

    return cli, fea_val_lable_dict, fea_emb_dim


def create_dataloader(df, lossname, modle, use_cuda):
    # 4以上转换为positive，其他转换为negative
    global tensor_x, tensor_y
    if lossname == 'auc':
        df['rating'] = np.where(df['rating'] >= 4, 1, 0)

    # 构建tensor
    if modle == 'mlp':
        df = df[["user_id", "movie_id", "rating"]]

    df_tensor = torch.tensor(df.values, dtype=torch.float32)
    if use_cuda:
        df_tensor = df_tensor.cuda()
    if modle == 'pnn':
        tensor_x, tensor_y = torch.split(df_tensor, [7, 1], dim=1)
    elif modle == 'mlp':
        tensor_x, tensor_y = torch.split(df_tensor, [2, 1], dim=1)
    elif modle == 'mynet':
        tensor_x, tensor_y = torch.split(df_tensor, [3, 1], dim=1)
    else:
        import warnings
        warnings.warn('fed modle wrong', UserWarning)

    # 根据Tensor创建数据集
    dataset = TensorDataset(tensor_x, tensor_y.view(-1))
    # 使用DataLoader加载数据集
    train_data = DataLoader(dataset, shuffle=True, batch_size=128, drop_last=True)
    return train_data  # , valid_data, test_data


def get_uv_cli_info(path):
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


def readdata_ML100k(path, lossname, modle, use_cuda, is_same_emb=True, istest=True):
    files = []
    for idx in range(1, 6):
        train_file = path + "train_" + str(idx) + ".csv"
        files.append(train_file)
    pa = path + "data_all.csv"
    all_data = pd.read_csv(pa)

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
    fea_emb_dims = []
    fea_value_label = []
    domain_id = 0
    for file in files:
        columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre', 'rating']
        # 读取转换好格式的数据
        df = pd.read_csv(file)  # , converters={'genre': literal_eval})
        df.columns = columns
        print("user_id: ", df["user_id"].nunique())

        df = df.apply(pd.to_numeric, errors='coerce')
        if istest:
            df = df[:5000]
        df_copy = copy.deepcopy(df)
        client_data.append(df_copy)
        if is_same_emb is False:
            df, fea_val_lable_dict, fea_emb_dim = get_fea_emb_dims_diff(df)
            fea_value_label.append(fea_val_lable_dict)
            fea_emb_dims.append(fea_emb_dim)
        # 分成训练集，验证集，测试集
        if modle == 'mynet':
            cols = ["user_id", "movie_id", "rating"]
            df = df[cols]
            cols.insert(0, 'domain_id')  # 在列索引为0的位置插入一列,列名为:domain_id，刚插入时不会有值，整列都是NaN
            df = df.reindex(columns=cols)
            df = df.fillna(domain_id)
            domain_id += 1
            # df.apply(lambda x: x.amount if x.list != "" else 0, axis=1)

        train_df = df.sample(frac=0.7, random_state=123, replace=False)
        other = df[~df.index.isin(train_df.index)]

        valid_df = other.sample(frac=0.4, random_state=123, replace=False)
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

    total_valid_datas = create_dataloader(total_valid_data, lossname, modle, use_cuda)
    total_test_datas = create_dataloader(total_test_data, lossname, modle, use_cuda)
    if is_same_emb:
        fea_emb_dims = get_fea_emb_dims_same(all_data)
    uv_cli_info = get_uv_cli_info(path)
    return train_datas, valid_datas, test_datas, total_valid_datas, total_test_datas, fea_emb_dims, uv_cli_info, client_data, fea_value_label


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

    def __init__(self, hidden_units, dropout=0.):
        """
        hidden_units:列表，
        每个元素表示每一层的神经单元个数，比如[256, 128, 64]，两层网络， 第一层神经单元128个，第二层64，第一个是输入维度
        """
        super(DNN, self).__init__()

        # nn.Linear输入是(输入特征数量， 输出特征数量)格式， 所以传入hidden_units，
        # Pytorch中线性层只有线性层， 不带激活函数。与tf里面的Dense不一样
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
    def __init__(self, emb_dims, hidden_units):
        super(MLP, self).__init__()
        Layers = [10, 64, 32, 16]
        self.embed_user_id = nn.Embedding(num_embeddings=emb_dims['user_id'], embedding_dim=10)
        self.embed_movie_id = nn.Embedding(num_embeddings=emb_dims['movie_id'], embedding_dim=10)
        self.dropout = 0
        """ fully connected layer """
        self.MLP_Layers = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(Layers[:-1], Layers[1:]))])

        # self.dense_final = nn.Linear(hidden_units[-1], 1)
        self.dense_final = nn.Linear(16, 1)

    def forward(self, inputs, lossname='mae'):
        """ 嵌入 """
        inputs = inputs.long()
        embed_user_id = self.embed_user_id(inputs[:, 0])
        embed_movie_id = self.embed_movie_id(inputs[:, 1])

        """ 拼接 """
        embedding_cat = torch.cat([embed_user_id, embed_movie_id], dim=-1)

        """ 点乘 """
        embedding_vec = torch.mul(embed_user_id, embed_movie_id)
        """ 全连接 """
        for mlp in self.MLP_Layers:
            embedding_vec = mlp(embedding_vec)
            embedding_vec = F.relu(embedding_vec)

        out = self.dense_final(embedding_vec)
        if lossname == 'mae':
            out = F.relu(out)
        outputs = out.view(-1)
        return outputs


class ProductLayer(nn.Module):

    def __init__(self, embed_dim, fea_num, hidden_units, use_cuda):
        super(ProductLayer, self).__init__()
        # 交叉分为两部分， z部分是单独的特征叠加，p部分是两两交叉
        # z部分的w， 这里的神经单元个数是hidden_units[0]=256
        self.w_z = nn.Parameter(torch.rand([fea_num, embed_dim, hidden_units[0]]))
        self.w_p = nn.Parameter(torch.rand([fea_num, fea_num, hidden_units[0]]))  # [26,26,256]
        self.l_b = torch.rand([hidden_units[0], ], requires_grad=True)
        if use_cuda:
            self.l_b = self.l_b.cuda()

    def forward(self, z, cross_embeds):
        # lz部分,
        # w_z:[7,10,64], w_p:[7, 7, 64],w_z.shape[2]=64;
        # z=cross_embeds:[128,7,10]
        # z.reshape(z.shape[0], -1)=[128,70];  w_z.permute(2,0,1)=[64,7,10].reshape(64,-1)=[64,70].T=[70,64]
        # l_z = torch.mm([128,70], [70,64])=[128,64]
        w_z = self.w_z.permute((2, 0, 1)).reshape(self.w_z.shape[2], -1).T
        l_z = torch.mm(z.reshape(z.shape[0], -1), w_z)  # (None, hidden_units[0])

        # lp 部分
        # 内积两两embedding内积得到的[field_dim, field_dim]的矩阵
        # matmul([128,7,10], [128, 10, 7])=[128, 7, 7]
        p = torch.matmul(cross_embeds, cross_embeds.permute((0, 2, 1)))  # [None, fea_num, fea_num]
        p = p.reshape(p.shape[0], -1)
        w_p = self.w_p.permute((2, 0, 1)).reshape(self.w_p.shape[2], -1).T
        # mm([128,49],[49,64])=[128,64]
        l_p = torch.mm(p, w_p)  # [None, hidden_units[0]]
        # output = [128,64]+[128,64]+[64]=[128,64]
        output = l_p + l_z + self.l_b
        return output


def get_one_layer(hidden_layer, weight_layer, bias_layer, cf_vec, int_size,
                  out_size):  # get one layer in MetaMF
    hid = hidden_layer(cf_vec)  # hid=[batch_size, hidden_size]
    hid = F.relu(hid)
    weight = weight_layer(hid)  # weight=[batch_size, self.layer_list[i-1]*self.layer_list[i]]
    bias = bias_layer(hid)  # bias=[batch_size, self.layer_list[i]]
    weight = weight.view(-1, int_size, out_size)
    bias = bias.view(-1, 1, out_size)
    return weight, bias


def get_item_embedding(hidden_layer, emb_layer_1, emb_layer_2, cf_vec, item_num, item_mem_num, item_emb_size):
    hid = hidden_layer(cf_vec)  # hid=[batch_size, hidden_size]
    hid = F.relu(hid)
    emb_left = emb_layer_1(hid)  # emb_left=[batch_size, item_num*item_mem_num]
    emb_right = emb_layer_2(hid)  # emb_right=[batch_size, item_mem_num*item_emb_size]
    emb_left = emb_left.view(-1, item_num, item_mem_num)  # emb_left=[batch_size, item_num, item_mem_num]
    emb_right = emb_right.view(-1, item_mem_num,
                               item_emb_size)  # emb_right=[batch_size, item_mem_num, item_emb_size]
    item_embedding = torch.matmul(emb_left, emb_right)  # item_embedding=[batch_size, item_num, item_emb_size]
    return item_embedding


def define_one_layer(mem_size, hidden_size, int_size, out_size):  # define one layer in MetaMF
    hidden_layer = nn.Linear(mem_size, hidden_size)
    weight_layer = nn.Linear(hidden_size, int_size * out_size)
    bias_layer = nn.Linear(hidden_size, out_size)
    return hidden_layer, weight_layer, bias_layer


def define_item_embedding(item_num, item_emb_size, item_mem_num, mem_size, hidden_size):
    hidden_layer = nn.Linear(mem_size, hidden_size)
    emb_layer_1 = nn.Linear(hidden_size, item_num * item_mem_num)
    emb_layer_2 = nn.Linear(hidden_size, item_mem_num * item_emb_size)
    return hidden_layer, emb_layer_1, emb_layer_2


path = "../input/domainsingle/"
path.split()


class DomainLayer(nn.Module):  # in fact, it's not a hypernetwork
    def __init__(self, domain_num, item_num, emb_dim, item_mem_num, mem_size, hidden_size):
        super(DomainLayer, self).__init__()
        self.item_num = item_num
        self.emb_dim = emb_dim
        self.item_mem_num = item_mem_num
        # For each user
        self.domain_embedding = nn.Embedding(domain_num, emb_dim)  # domain_embedding融合其他domain的信息
        self.shared_memory = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(emb_dim, mem_size)), requires_grad=True)

        self.hidden_layer_3, self.emb_layer_1, self.emb_layer_2 = define_item_embedding(item_num, emb_dim,
                                                                                        item_mem_num, mem_size,
                                                                                        hidden_size)

    def forward(self, domain_id):
        domain_emb = self.domain_embedding(domain_id)  # input_user=[batch_size, user_emb_size]
        domain_cf_vec = torch.matmul(domain_emb, self.shared_memory)  # cf_vec=[batch_size, mem_size]
        item_embedding = get_item_embedding(self.hidden_layer_3, self.emb_layer_1, self.emb_layer_2, domain_cf_vec,
                                            self.item_num, self.item_mem_num, self.emb_dim)
        return item_embedding


# PNN网络
# 逻辑是底层输入（类别型特征) -> embedding层 -> product 层 -> DNN -> 输出
class myNet(nn.Module):
    # hidden_units = [256, 128, 64]
    def __init__(self, inputs_info, hidden_units, use_cuda, dnn_dropout=0., embed_dim=10, outdim=1):
        super(myNet, self).__init__()
        # self.features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']
        self.iputs_info = inputs_info
        self.user_embed = nn.Embedding(num_embeddings=inputs_info['user_id'], embedding_dim=embed_dim)
        # self.embed_movie_id = nn.Embedding(num_embeddings=inputs_info['movie_id'], embedding_dim=embed_dim)
        self.domain = DomainLayer(domain_num=inputs_info['domain_id'],
                                  item_num=inputs_info['movie_id'],
                                  emb_dim=embed_dim, item_mem_num=8, mem_size=64, hidden_size=128)
        # cross层
        self.product = ProductLayer(embed_dim, len(inputs_info), hidden_units, use_cuda)
        # dnn 层
        self.dnn_network = DNN(hidden_units, dnn_dropout)
        self.dense_final = nn.Linear(hidden_units[-1], 1)

    def forward(self, inputs, Debug=False, lossname='mae'):
        # features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']
        inputs = inputs.long()
        # inputs = ["domain_id", user_id", "movie_id", "rating"]
        user_embed = self.user_embed(inputs[:, 1])
        item_embeddings = self.domain(domain_id=inputs[:, 0])
        item_id = inputs[:, 2].view(-1, 1)  # item_id=[batch_size, 1]
        item_one_hot = torch.zeros(len(item_id), self.iputs_info['movie_id'],
                                   device=item_id.device)  # we generate it dynamically, and default device is cpu
        item_one_hot.scatter_(1, item_id, 1)  # item_one_hot=[batch_size, item_num]
        item_one_hot = torch.unsqueeze(item_one_hot, 1)  # item_one_hot=[batch_size, 1, item_num]
        item_embed = torch.matmul(item_one_hot, item_embeddings)  # out=[batch_size, 1, item_emb_size]
        item_embed = item_embed.squeeze()
        # product layer foward
        sparse_embeds = []
        sparse_embeds.append(user_embed)
        sparse_embeds.append(item_embed)
        sparse_embeds = torch.stack(sparse_embeds)  # [fea_num, batch_sz, emb_dim]->[7,128,10]
        # [None, sparse_num, embed_dim]  注意此时空间不连续， 下面改变形状不能用view，用reshape
        sparse_embeds = sparse_embeds.permute((1, 0, 2))  # [batch_sz, fea_num, emb_dim]->[128,7,10]
        z = sparse_embeds
        inputs = self.product(z, sparse_embeds)
        l1 = F.relu(inputs)
        dnn_x = self.dnn_network(l1)
        # outputs = torch.sigmoid(self.dense_final(dnn_x))
        outputs = self.dense_final(dnn_x)
        if lossname == 'mae':
            outputs = F.relu(outputs)
        outputs = outputs.view(-1)
        return outputs


# PNN网络
# 逻辑是底层输入（类别型特征) -> embedding层 -> product 层 -> DNN -> 输出
class PNN(nn.Module):
    # hidden_units = [256, 128, 64]
    def __init__(self, emb_info, hidden_units, use_cuda, mode_type='in', dnn_dropout=0., embed_dim=10, outdim=1):
        super(PNN, self).__init__()
        self.features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']
        self.emb_info = emb_info
        self.fea_num = len(self.features)  # 26 C
        self.mode_type = mode_type
        self.embed_dim = embed_dim
        # embedding层，类别特征embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=self.embed_dim)
            for key, val in self.emb_info.items()
        })
        # Product层
        self.prodcut = ProductLayer(mode_type, embed_dim, self.fea_num, hidden_units, use_cuda)
        # dnn 层
        # hidden_units[0] += self.dense_num  # dense_inputs直接输入道dnn，没有embedding 256+13=269
        self.dnn_network = DNN(hidden_units, dnn_dropout)
        self.dense_final = nn.Linear(hidden_units[-1], 1)

    def forward(self, x, Debug=False, lossname='mae'):
        # features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']
        inputs = x.long()
        cross_embeds = [self.embed_layers['embed_' + key](inputs[:, i])
                        for key, i in zip(self.emb_info.keys(), range(inputs.shape[1]))]

        # len(cross_embeds)=7, [fea_num, None, embed_dim]->(7,none,10)
        cross_embeds = torch.stack(cross_embeds)  # [fea_num, batch_sz, emb_dim]->[7,128,10]
        # [None, fea_num, embed_dim] 此时空间不连续， 下面改变形状不能用view，用reshape
        cross_embeds = cross_embeds.permute((1, 0, 2))  # [batch_sz, fea_num, emb_dim]->[128,7,10]
        z = cross_embeds  # [128,7,10]

        # product layer foward
        inputs = self.prodcut(z, cross_embeds)
        l1 = F.relu(inputs)
        dnn_x = self.dnn_network(l1)
        # outputs = torch.sigmoid(self.dense_final(dnn_x))
        outputs = self.dense_final(dnn_x)
        if lossname == 'mae':
            outputs = F.relu(outputs)
        outputs = outputs.view(-1)
        return outputs


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
def get_auc(y_true, y_pred):
    y_pred = y_pred.data.detach().cpu().numpy()
    y_true = y_true.data.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Embedding') != -1:
        torch.nn.init.normal_(m.weight.data, std=0.01)
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)


# torchkeras
def init_net(cli_nums, use_cuda, same_dim, fea_emb_dims, modletype):
    global net
    import torchkeras
    hidden_units = [64, 32, 16]
    local_nets = []
    # 模型初始化
    for i in range(cli_nums):
        if same_dim:
            emb_dims = fea_emb_dims
        else:
            emb_dims = fea_emb_dims[i]
        if modletype == 'mlp':
            net = MLP(emb_dims, hidden_units)
        elif modletype == 'pnn':
            net = PNN(emb_dims, hidden_units, use_cuda)
        elif modletype == 'mynet':
            fea_emb_dims['domain_id'] = cli_nums
            net = myNet(emb_dims, hidden_units, use_cuda)
        else:
            import warnings
            warnings.warn('modletype argument wrong', UserWarning)
        if use_cuda:
            net.cuda()
        net = torchkeras.Model(net)
        print(net)
        net.apply(weights_init)
        local_nets.append(net)
    return local_nets


def fed_torchkeras(T, fed_type, train_datas, valid_datas, test_datas, total_valid_datas, total_test_datas, fea_emb_dims,
                   uv_cli_info, client_data, fea_val_label, use_cuda, modletype, same_dim=True, test_type="together",
                   lossname='auc'):
    cli_nums = len(train_datas)
    local_nets = init_net(cli_nums, use_cuda, same_dim, fea_emb_dims, modletype)
    # 全部客户端的metrics记录格式
    # cols = ["epoch", "loss", "auc", "val_loss", "val_auc", "fed_loss", "fed_auc"]
    cols = ["epoch",
            "loss", "mae", "mse", "rmse",
            "val_loss", "val_mae", "val_mse", "val_rmse",
            "fed_loss", "fed_mae", "fed_mse", "fed_rmse"]
    df_history = pd.DataFrame(columns=cols)

    # 联邦训练
    epochs = [1, 2, 1, 4, 2]
    lr = 0.001
    for t in range(T):
        metric_fed_befroe = pd.DataFrame()
        for idx in range(cli_nums):
            print("-----------------------------------------------------------client", idx + 1, modletype, fed_type,
                  "fed", t + 1)
            if (t + 1) % 15 == 0:
                lr = round(lr * np.power(0.97, (T / 10)), 4)
            print("decay_lr: ", lr)
            # 参数
            local_nets[idx].compile(loss_func=torch.nn.MSELoss(),
                                    optimizer=torch.optim.Adam(params=local_nets[idx].parameters(), lr=lr,
                                                               weight_decay=0.001),
                                    metrics_dict={"mae": get_mae, "mse": get_mse, "rmse": get_rmse})
            # 训练
            met = local_nets[idx].fit(epochs=1, dl_train=train_datas[idx], dl_val=valid_datas[idx],
                                      log_step_freq=3000)
            # 拼接 epoch | loss |  mae  | mse | rmse |  val_mae  | val_mse | val_rmse
            metric_fed_befroe = pd.concat([metric_fed_befroe, met], axis=0)
        # 联邦之前各客户端平均指标
        metric_fed_befroe = metric_fed_befroe.mean(axis=0)
        # 联邦聚合
        if fed_type == "FedCross":
            update_user_emb_avg(local_nets, client_data, uv_cli_info, fea_val_label, modletype, use_cuda,
                                same_dim)
            update_item_emb_avg(local_nets, client_data, uv_cli_info, fea_val_label, modletype, use_cuda,
                                same_dim)
            # local_nets = average_other_layers(local_nets)
        elif fed_type == "FedAvg":
            average_layers(local_nets)
        else:
            import warnings
            warnings.warn('fed argument wrong', UserWarning)
        # 联邦之后的平均指标
        metric_fed_after = pd.DataFrame()
        for idx in range(cli_nums):
            # 客户端评估指标
            fed_met = local_nets[idx].evaluate(valid_datas[idx])  # 返回的是字典
            fed_met = pd.DataFrame([fed_met])
            metric_fed_after = pd.concat([metric_fed_after, fed_met], axis=0)
        # 计算客户端的平均指标
        print("metric_fed_befroe \n", metric_fed_befroe)
        metric_fed_after = metric_fed_after.mean(axis=0)
        print("metric_fed_after \n", metric_fed_after)
        # 记录
        history = (t + 1,
                   metric_fed_befroe["loss"], metric_fed_befroe["mae"], metric_fed_befroe["mae"],
                   metric_fed_befroe["rmse"],
                   metric_fed_befroe["val_loss"], metric_fed_befroe["val_mae"], metric_fed_befroe["val_mse"],
                   metric_fed_befroe["val_rmse"],
                   metric_fed_after["val_loss"], metric_fed_after["val_mae"], metric_fed_after["val_mse"],
                   metric_fed_after["val_rmse"])
        df_history.loc[t] = history

    return df_history


def train_fed():
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    path = '../dataset/preprocessed_data/ml-100k/non_iid/'
    is_test = False
    modletype = 'mynet'
    is_same_emb = True
    lossname = 'mae'
    test_type = 'single'
    T = 2
    # 读取数据
    train_datas, valid_datas, test_datas, total_valid_datas, total_test_datas, fea_emb_dims, uv_cli_info, client_data, fea_value_label = readdata_ML100k(
        path, lossname, modletype, use_cuda, is_same_emb=is_same_emb, istest=is_test)
    fed_type = "FedCross"
    fed_type = "FedAvg"

    df_history = fed_torchkeras(T, fed_type, train_datas, valid_datas, test_datas, total_valid_datas, total_test_datas,
                                fea_emb_dims, uv_cli_info, client_data, fea_value_label, use_cuda, modletype,
                                same_dim=is_same_emb, test_type=test_type,
                                lossname=lossname)
    plot_metric2(df_history, fed_type, is_test=is_test)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")


train_fed()

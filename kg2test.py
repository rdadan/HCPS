import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

"=================================================================================================================================================ulitiy"


def update_domain_emb_avg(nets, use_cuda):
    embs_d = []

    name_d = 'domain.shared_memory_d'
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
    # 构建tensor
    if modle == 'mlp':
        df = df[["user_id", "movie_id", "rating"]]

    df_tensor = torch.tensor(df.values, dtype=torch.float32)
    if use_cuda:
        df_tensor = df_tensor.cuda()
    if modle == 'pnn':
        tensor_x, tensor_y = torch.split(df_tensor, [7, 1], dim=1)
    elif modle == 'mlp' or modle == 'metamf':
        tensor_x, tensor_y = torch.split(df_tensor, [2, 1], dim=1)
    elif modle == 'mynet':
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


def readdata_ML100k(path, lossname, modle, use_cuda, is_same_emb=True, istest=True, file=None):
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
    total_len = 0
    for file in files:
        columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre', 'rating']
        # 读取转换好格式的数据
        df = pd.read_csv(file)  # , converters={'genre': literal_eval})
        df.columns = columns
        if lossname == 'auc':
            df['rating'] = np.where(df['rating'] > 3, 1, 0)
        print("read file: ", file, len(df))
        total_len += len(df)
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
        if modle == 'mynet' or modle == 'metamf':
            cols = ["user_id", "movie_id", "rating"]
            df = df[cols]
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
        fea_emb_dims = get_fea_emb_dims_same(all_data)
    # uv_cli_info = read_uv_cli_info(path)
    uv_cli_info = get_uv_cli_info(all_data, client_data)

    return train_datas, valid_datas, test_datas, total_valid_datas, total_test_datas, fea_emb_dims, uv_cli_info, client_data, fea_value_label


def readdata_ML100k_cnetral(path, lossname, modle, use_cuda, istest=True):
    pa = path + "data_all.csv"
    all_data = pd.read_csv(pa)
    all_data = all_data.apply(pd.to_numeric, errors='coerce')
    if istest:
        all_data = all_data[:5000]
    features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']
    fed_emb_infos = {}
    for fea in features:
        fed_emb_infos[fea] = all_data[fea].nunique()

    # 分成训练集，验证集，测试集
    train = all_data.sample(frac=0.7, random_state=123, replace=False)
    other = all_data[~all_data.index.isin(train.index)]
    valid = other.sample(frac=0.4, random_state=123, replace=False)
    test = other[~other.index.isin(valid.index)]
    # DataLoader加载
    train_data = create_dataloader(train, lossname, modle, use_cuda)
    valid_data = create_dataloader(valid, lossname, modle, use_cuda)
    test_data = create_dataloader(test, lossname, modle, use_cuda)
    return train_data, valid_data, test_data, fed_emb_infos


"=================================================================================================================================================net"

import torch.nn as nn
import os

os.environ['CUDA_ENABLE_DEVICES'] = '0'


# 定义一个全连接层的神经网络
class DNN(nn.Module):

    def __init__(self, hidden_units, dropout=0.):
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

    def __init__(self, mode_type, embed_dim, sparse_num, hidden_units, use_cuda):
        super(ProductLayer, self).__init__()
        self.mode_type = mode_type
        self.w_z = nn.Parameter(torch.rand([sparse_num, embed_dim, hidden_units[0]]))
        # p部分, 分内积和外积两种操作
        if mode_type == 'in':
            self.w_p = nn.Parameter(torch.rand([sparse_num, sparse_num, hidden_units[0]]))  # [26,26,256]
        else:
            self.w_p = nn.Parameter(torch.rand([embed_dim, embed_dim, hidden_units[0]]))  # [10,10,256]

        self.l_b = torch.rand([hidden_units[0], ], requires_grad=True)
        if use_cuda:
            self.l_b = self.l_b.cuda()

    def forward(self, z, sparse_embeds):
        # lz部分,
        # w_z:[7,10,64], w_p:[7, 7, 64],w_z.shape[2]=64;
        # z=sparse_embeds:[128,7,10]
        # z.reshape(z.shape[0], -1)=[128,70];  w_z.permute(2,0,1)=[64,7,10].reshape(64,-1)=[64,70].T=[70,64]
        # l_z = torch.mm([128,70], [70,64])=[128,64]
        l_z = torch.mm(z.reshape(z.shape[0], -1),
                       self.w_z.permute((2, 0, 1)).reshape(self.w_z.shape[2], -1).T)  # (None, hidden_units[0])

        # lp 部分
        # in模式  内积操作  p就是两两embedding先内积得到的[field_dim, field_dim]的矩阵
        if self.mode_type == 'in':
            # matmul可处理维度不同的矩阵, : [2,5,3]*[1,3,4]->[2,5,4]
            # matmul([128,7,10], [128, 10, 7])=[128, 7, 7]
            p = torch.matmul(sparse_embeds, sparse_embeds.permute((0, 2, 1)))  # [None, sparse_num, sparse_num]
        # 外积模式  这里的p矩阵是两两embedding先外积得到n*n个[embed_dim, embed_dim]的矩阵， 然后对应位置求和得到最终的1个[embed_dim, embed_dim]的矩阵
        # 所以这里实现的时候， 可以先把sparse_embeds矩阵在sparse_num方向上先求和， 然后再外积
        else:
            f_sum = torch.unsqueeze(torch.sum(sparse_embeds, dim=1), dim=1)  # [None, 1, embed_dim]
            p = torch.matmul(f_sum.permute((0, 2, 1)), f_sum)  # [None, embed_dim, embed_dim]
        # mm([128,49],[49,64])=[128,64]
        l_p = torch.mm(p.reshape(p.shape[0], -1),
                       self.w_p.permute((2, 0, 1)).reshape(self.w_p.shape[2], -1).T)  # [None, hidden_units[0]]
        # output = [128,64]+[128,64]+[64]=[128,64]
        output = l_p + l_z + self.l_b
        return output


# PNN网络
# 逻辑是底层输入（类别型特征) -> embedding层 -> product 层 -> DNN -> 输出
class PNN(nn.Module):
    # hidden_units = [256, 128, 64]
    def __init__(self, emb_info, hidden_units, use_cuda, mode_type='in', dnn_dropout=0., embed_dim=10, outdim=1):
        super(PNN, self).__init__()
        self.features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']
        self.emb_info = emb_info
        # self.dense_num = len(self.dense_feas)  # 13 L
        self.sparse_num = len(self.features)  # 26 C
        self.mode_type = mode_type
        self.embed_dim = embed_dim

        # embedding层，类别特征embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=self.embed_dim)
            for key, val in self.emb_info.items()
        })

        # Product层
        self.product = ProductLayer(mode_type, embed_dim, self.sparse_num, hidden_units, use_cuda)

        # dnn 层
        # hidden_units[0] += self.dense_num  # dense_inputs直接输入道dnn，没有embedding 256+13=269
        self.dnn_network = DNN(hidden_units, dnn_dropout)
        self.dense_final = nn.Linear(hidden_units[-1], 1)

    def forward(self, x, Debug=False, lossname='mae'):
        # features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'release_date', 'genre']
        inputs = x.long()
        sparse_embeds = []

        # sparse_embeds = [self.embed_layers['embed_' + key](inputs[:, i])
        #                  for key, i in zip(self.emb_info.keys(), range(inputs.shape[1]))]
        user_emb = self.embed_layers['embed_user_id'](inputs[:, 0])
        sparse_embeds.append(user_emb)

        age_emb = self.embed_layers['embed_age'](inputs[:, 1])
        sparse_embeds.append(age_emb)

        gender_emb = self.embed_layers['embed_gender'](inputs[:, 2])
        sparse_embeds.append(gender_emb)

        occupation_emb = self.embed_layers['embed_occupation'](inputs[:, 3])
        sparse_embeds.append(occupation_emb)

        movie_emb = self.embed_layers['embed_zip_code'](inputs[:, 4])
        sparse_embeds.append(movie_emb)

        movie_emb = self.embed_layers['embed_movie_id'](inputs[:, 5])
        sparse_embeds.append(movie_emb)

        # movie_emb = self.embed_layers['embed_release_date'](inputs[:, 6])
        # sparse_embeds.append(movie_emb)

        movie_emb = self.embed_layers['embed_genre'](inputs[:, 6])
        sparse_embeds.append(movie_emb)

        # len(sparse_embeds)=7, [fea_num, None, embed_dim]->(7,none,10)
        sparse_embeds = torch.stack(sparse_embeds)  # [fea_num, batch_sz, emb_dim]->[7,128,10]
        # [None, sparse_num, embed_dim]  注意此时空间不连续， 下面改变形状不能用view，用reshape
        sparse_embeds = sparse_embeds.permute((1, 0, 2))  # [batch_sz, fea_num, emb_dim]->[128,7,10]
        z = sparse_embeds  # [128,7,10]

        # product layer foward
        inputs = self.product(z, sparse_embeds)
        # 加数值特征
        # l1 = F.relu(torch.cat([inputs, dense_inputs], -1))
        l1 = F.relu(inputs)
        # dnn_network
        dnn_x = self.dnn_network(l1)
        # outputs = torch.sigmoid(self.dense_final(dnn_x))
        outputs = self.dense_final(dnn_x)
        if lossname == 'mae':
            outputs = F.relu(outputs)
        outputs = outputs.view(-1)
        return outputs


def set_uv_emb_layer(item_num, item_emb_size, item_mem_num, mem_size, hidden_size):
    hidden_layer = nn.Linear(mem_size, hidden_size)
    emb_layer_1 = nn.Linear(hidden_size, item_num * item_mem_num)
    emb_layer_2 = nn.Linear(hidden_size, item_mem_num * item_emb_size)
    return hidden_layer, emb_layer_1, emb_layer_2


def get_uv_embedding(hidden_layer, emb_layer_1, emb_layer_2, cf_vec, item_num, item_mem_num, item_emb_size):
    hid = hidden_layer(cf_vec)  # hid=[batch_size, hidden_size]
    hid = F.relu(hid)
    emb_left = emb_layer_1(hid)  # emb_left=[batch_size, item_num*item_mem_num]
    emb_right = emb_layer_2(hid)  # emb_right=[batch_size, item_mem_num*item_emb_size]
    emb_left = emb_left.view(-1, item_num, item_mem_num)  # emb_left=[batch_size, item_num, item_mem_num]
    emb_right = emb_right.view(-1, item_mem_num, item_emb_size)  # emb_right=[batch_size, item_mem_num, item_emb_size]
    item_embedding = torch.matmul(emb_left, emb_right)  # item_embedding=[batch_size, item_num, item_emb_size]
    return item_embedding


class DomainLayer(nn.Module):  # in fact, it's not a hypernetwork
    def __init__(self, domain_num, emb_dim, mem_num, mem_size, hidden_size):
        super(DomainLayer, self).__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.mem_num = mem_num
        # For each user
        self.domain_embedding = nn.Embedding(self.domain_num, emb_dim)  # domain_embedding融合其他domain的信息
        self.shared_memory_d = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(emb_dim, mem_size)), requires_grad=True)
        self.item_hidden_layer, self.item_emb_l1, self.item_emb_l2 = set_uv_emb_layer(domain_num, emb_dim, mem_num,
                                                                                      mem_size, hidden_size)

    def forward(self, domain_id):
        domain_emb = self.domain_embedding(domain_id)
        domain_cf_vec_d = torch.matmul(domain_emb, self.shared_memory_d)  # [1,10]*[10,64]=[1,10]
        d_embs = get_uv_embedding(self.item_hidden_layer, self.item_emb_l1, self.item_emb_l2, domain_cf_vec_d,
                                  self.domain_num, self.mem_num, self.emb_dim)  # 【1，18，10】
        return d_embs.squeeze(0)


class myNet(nn.Module):
    # hidden_units = [256, 128, 64]
    def __init__(self, inputs_info, domain_id, uv_domains_info, hidden_units, use_cuda, dnn_dropout=0., embed_dim=10,
                 outdim=1):
        super(myNet, self).__init__()
        # self.features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']

        self.use_cuda = use_cuda
        self.iputs_info = inputs_info
        self.domain_id = domain_id
        self.uv_domains_info = uv_domains_info
        self.embed_user_id = nn.Embedding(num_embeddings=inputs_info['user_id'], embedding_dim=embed_dim)
        self.embed_movie_id = nn.Embedding(num_embeddings=inputs_info['movie_id'], embedding_dim=embed_dim)
        # 共享域
        self.domain = DomainLayer(domain_num=inputs_info['domain_id'], emb_dim=embed_dim, mem_num=10, mem_size=64,
                                  hidden_size=64)
        # cross层
        self.product = ProductLayer('in', embed_dim, 4, hidden_units, use_cuda)
        # dnn 层
        self.dnn_network = DNN(hidden_units, dnn_dropout)
        self.dense_final = nn.Linear(hidden_units[-1], 1)

    def forward(self, inputs, lossname='mae'):
        # features = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre']
        # inputs = ["user_id", "movie_id", "rating"]
        inputs = inputs.long()
        # 域
        share_domain_embs = self.domain(self.domain_id)
        # at user space
        user_domain_avgemb = self.get_uv_domain_avgemb(share_domain_embs, inputs[:, 0], self.uv_domains_info[0])
        item_domain_avgemb = self.get_uv_domain_avgemb(share_domain_embs, inputs[:, 1], self.uv_domains_info[1])
        user_emb = self.embed_user_id(inputs[:, 0])
        item_emb = self.embed_movie_id(inputs[:, 1])
        # 交叉
        sparse_embeds = []
        sparse_embeds.append(user_emb)
        sparse_embeds.append(item_emb)
        sparse_embeds.append(user_domain_avgemb)
        sparse_embeds.append(item_domain_avgemb)
        sparse_embeds = torch.stack(sparse_embeds)  # [fea_num, batch_sz, emb_dim]->[7,128,10]
        # [None, sparse_num, embed_dim]  改变形状不能用view，用reshape
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

    def get_emb(self, id, num, emb):
        id = id.view(-1, 1)
        one_hot = torch.zeros(len(id), num, device=id.device)  # we generate it dynamically, and default device is cpu
        one_hot.scatter_(1, id, 1)  # item_one_hot=[batch_size, item_num]
        one_hot = torch.unsqueeze(one_hot, 1)  # item_one_hot=[batch_size, 1, item_num]
        embed = torch.matmul(one_hot, emb)  # out=[batch_size, 1, item_emb_size]
        embed = embed.squeeze()
        return embed

    def get_uv_domain_avgemb(self, share_domain_embs, input_ids, uv_domains):
        domain_avgemb = []
        input_ids = input_ids.data.detach().cpu().numpy()
        for id in input_ids:
            domains = uv_domains[id]
            embs = []
            for d in domains:
                if self.use_cuda:
                    embs.append(share_domain_embs[d].detach().cpu().numpy())
                else:
                    embs.append(share_domain_embs[d].detach().numpy())
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


def mae_loss():
    return torch.nn.MSELoss()


def auc_loss(y_pred, y_true):
    return torch.nn.BCELoss()(y_pred, y_true)


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


# !/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter


# In[ ]:


class MetaRecommender(nn.Module):  # in fact, it's not a hypernetwork
    def __init__(self, user_num, item_num, item_emb_size=32, item_mem_num=8, user_emb_size=32, mem_size=64,
                 hidden_size=64):  # note that we have many users and each user has many layers
        super(MetaRecommender, self).__init__()
        self.item_num = item_num
        self.item_emb_size = item_emb_size
        self.item_mem_num = item_mem_num
        # For each user
        self.user_embedding = nn.Embedding(user_num, user_emb_size)
        self.memory = Parameter(nn.init.xavier_normal_(torch.Tensor(user_emb_size, mem_size)), requires_grad=True)
        # For each layer
        self.hidden_layer_1, self.weight_layer_1, self.bias_layer_1 = self.define_one_layer(mem_size, hidden_size,
                                                                                            item_emb_size,
                                                                                            int(item_emb_size / 4))
        self.hidden_layer_2, self.weight_layer_2, self.bias_layer_2 = self.define_one_layer(mem_size, hidden_size,
                                                                                            int(item_emb_size / 4), 1)
        self.hidden_layer_3, self.emb_layer_1, self.emb_layer_2 = self.define_item_embedding(item_num, item_emb_size,
                                                                                             item_mem_num, mem_size,
                                                                                             hidden_size)

    def define_one_layer(self, mem_size, hidden_size, int_size, out_size):  # define one layer in MetaMF
        hidden_layer = nn.Linear(mem_size, hidden_size)
        weight_layer = nn.Linear(hidden_size, int_size * out_size)
        bias_layer = nn.Linear(hidden_size, out_size)
        return hidden_layer, weight_layer, bias_layer

    def define_item_embedding(self, item_num, item_emb_size, item_mem_num, mem_size, hidden_size):
        hidden_layer = nn.Linear(mem_size, hidden_size)
        emb_layer_1 = nn.Linear(hidden_size, item_num * item_mem_num)
        emb_layer_2 = nn.Linear(hidden_size, item_mem_num * item_emb_size)
        return hidden_layer, emb_layer_1, emb_layer_2

    def forward(self, user_id):
        # collaborative memory module
        user_emb = self.user_embedding(user_id)  # input_user=[batch_size, user_emb_size]
        cf_vec = torch.matmul(user_emb,
                              self.memory)  # [user_num, u_emb_size]*[u_emb_size, mem_size]=[batch_size, mem_size]
        # collaborative memory module
        # collaborative memory module
        # meta recommender module
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
        return output_weight, output_bias, item_embedding, cf_vec  # ([len(layer_list)+1, batch_size, *, *], [len(layer_list)+1, batch_size, 1, *], [batch_size, item_num, item_emb_size], [batch_size, mem_size])

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


# In[ ]:


class MetaMF(nn.Module):
    def __init__(self, inputs_info, item_emb_size=10, item_mem_num=16, user_emb_size=10, mem_size=32,
                 hidden_size=32):
        super(MetaMF, self).__init__()
        self.item_num = inputs_info['movie_id']
        self.metarecommender = MetaRecommender(inputs_info['user_id'], inputs_info['movie_id'], item_emb_size,
                                               item_mem_num, user_emb_size, mem_size,
                                               hidden_size)

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
        out = torch.nn.Sigmoid()(out)
        out = torch.squeeze(out)  # out=[batch_size]
        out = out.view(-1)
        # prediction module
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Embedding') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)
    if classname.find("DomainLayer") != -1 or classname.find("ProductLayer") != -1:
        for key, val in m.state_dict().items():
            if key.find("bias") != -1:
                torch.nn.init.constant_(val, 0)
            else:
                torch.nn.init.xavier_normal_(val)


def init_net2(cli_nums, v_clis, use_cuda, same_dim, fea_emb_dims, modletype):
    global net
    # import torchkeras
    hidden_units = [64, 32, 16]
    local_nets = []
    for i in range(cli_nums):
        if same_dim:
            emb_dims = fea_emb_dims
        else:
            emb_dims = fea_emb_dims[i]
        if modletype == 'mlp':
            print("mlp")
            net = MLP(emb_dims, hidden_units)
        elif modletype == 'metamf':
            print("metamf")
            net = MetaMF(emb_dims)
        elif modletype == 'pnn':
            print("pnn")
            net = PNN(emb_dims, hidden_units, use_cuda)
        elif modletype == 'mynet':
            domain_id = torch.tensor(i).long()
            if use_cuda:
                domain_id = domain_id.cuda()
            fea_emb_dims['domain_id'] = cli_nums
            net = myNet(emb_dims, domain_id, v_clis, hidden_units, use_cuda)
            # print(net)
        else:
            import warnings
            warnings.warn('modletype argument wrong', UserWarning)
        if use_cuda:
            net.cuda()
        # print(net)
        net.apply(weights_init)
        # net = torchkeras.Model(net)
        local_nets.append(net)
    return local_nets


def do_train(T, fed_type, train_datas, valid_datas, test_datas, total_valid_datas, total_test_datas, fea_emb_dims,
             uv_cli_info, client_data, fea_val_label, use_cuda, modletype, same_dim=True, test_type="together",
             lossname='auc'):
    cli_nums = len(train_datas)
    local_nets = init_net2(cli_nums, uv_cli_info, use_cuda, same_dim, fea_emb_dims, modletype)
    # 联邦训练
    lr = 0.001
    for t in range(T):
        avgloss = 0
        avgauc = 0
        vavgloss = 0
        vavgauc = 0
        for idx in range(cli_nums):
            print(
                "------------------------------------------------------------------------------------------------client",
                idx + 1, modletype, fed_type, "t", t + 1)
            if (t + 1) % 4 == 0:
                lr = round(lr * np.power(0.97, (T / 4)), 4)
            local_nets[idx].train()
            train_loss = 0.0
            train_auc = 0.0
            optimizer = torch.optim.Adam(params=local_nets[idx].parameters(), lr=lr, weight_decay=0.001)
            step = 0
            for (features, y_true) in train_datas[idx]:
                step += 1
                y_pred = local_nets[idx](features)
                loss = torch.nn.BCELoss()(y_pred, y_true)
                mae = get_auc(y_pred, y_true)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_nets[idx].parameters(), 5)
                optimizer.step()
                train_loss += loss.item()
                train_auc += mae.item()
                if step % 500 == 0:
                    print("step: %s train loss: %.4f  train_auc:%.4f" % (step, train_loss / step, train_auc / step))

            train_loss = train_loss / step
            train_auc = train_auc / step
            avgloss += train_loss
            avgauc += train_auc
            print("train loss: %.4f  train_auc:%.4f" % (train_loss, train_auc))
            local_nets[idx].eval()
            valid_loss = 0.0
            valid_auc = 0.0
            step = 1
            for (features, y_true) in valid_datas[idx]:
                step += 1
                with torch.no_grad():
                    y_pred = local_nets[idx](features)
                    loss = auc_loss(y_pred, y_true)
                    mae = get_auc(y_pred, y_true)
                valid_loss += loss.item()
                valid_auc += mae.item()
            valid_loss = valid_loss / step
            valid_auc = valid_auc / step
            vavgloss += valid_loss
            vavgauc += valid_auc
            print("valid loss: %.4f  valid auc:%.4f" % (valid_loss, valid_auc))

        print("\n")
        print("train avg loss: %4.4f, train avg auc: %4.4f" % (avgloss / cli_nums, avgauc / cli_nums))
        print("valid avg loss: %4.4f, valid avg auc: %4.4f" % (vavgloss / cli_nums, vavgauc / cli_nums))

    return


def train_web2():
    use_cuda = torch.cuda.is_available()
    use_cuda = False
    is_test = False

    print(use_cuda)
    test_type = 'single'
    is_same_emb = True

    path1 = '../dataset/preprocessed_data/ml-100k/non_iid/'
    path2 = '../dataset/preprocessed_data/ml-1m/domain_multiple/'
    path3 = "../input/domainsingle/"
    path4 = "../input/domainmultiple/"

    fed_type_avg = "FedAvg"
    fed_type_cross = "FedCross"
    fed_type_central = "Central"
    fed_type_domain = "FedDomain"

    # 读取数据

    files = ['Children', 'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
             'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
             'Thriller', 'War', 'Western']
    file_5 = ['Drama', 'Comedy', 'Action', 'Thriller', 'Sci-Fi']
    file_10 = ['Drama', 'Comedy', 'Action', 'Thriller', 'Sci-Fi', 'Romance', 'Adventure', 'Crime', 'Children', 'War']
    file_15 = ['Drama', 'Comedy', 'Action', 'Thriller', 'Sci-Fi', 'Romance', 'Adventure', 'Crime', 'Children', 'War',
               'Horror', 'Animation', 'Mystery', 'Musical', 'Fantasy']
    file_2 = ['Film-Noir', 'Drama', 'Western']
    path = path2
    files = file_2
    lossname = 'auc'
    crossdomain = 'torchcat'
    modletype = 'metamf'
    fed_type = fed_type_domain
    T = 4

    train_datas, valid_datas, test_datas, total_valid_datas, total_test_datas, fea_emb_dims, uv_cli_info, client_data, fea_value_label = readdata_ML100k(
        path, lossname, modletype, use_cuda, is_same_emb=is_same_emb, istest=is_test, file=files)

    do_train(T, fed_type, train_datas, valid_datas, test_datas, total_valid_datas, total_test_datas,
             fea_emb_dims, uv_cli_info, client_data, fea_value_label,
             use_cuda, modletype, same_dim=is_same_emb, test_type=test_type,
             lossname=lossname)

    print("save net done xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")


train_web2()

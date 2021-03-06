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


from sklearn.decomposition import PCA


def get_pca_emb(embs):
    emb = np.array(embs).T
    pca = PCA(n_components=1)
    pca.fit(emb)
    emb = torch.tensor(pca.transform(emb).T).squeeze(0)
    return emb


##Python??????PCA
def my_pca(embs):  # k is the components you want
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
    eig_pairs.sort(reverse=False)
    # select the top k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:1]])
    # get new data
    data = np.dot(norm_X, np.transpose(feature))
    emb = torch.tensor(data.T).squeeze(0)
    return emb


def get_avg_emb(embs):
    avg_emb = torch.mean(torch.tensor(embs), dim=0, keepdim=True).squeeze(0)
    return avg_emb


def get_agg_emb(nets, clis, fea, fea_val, emb_name, fea_val_label, use_cuda, same_dim, get_emb):
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
    agg_emb = get_emb(embs)
    return agg_emb, fea_labels


def update_user_emb_avg(nets, clientsdata, uv_cli_infos, fea_val_label, modletype, use_cuda, same_dim):
    # clientsdata columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre', 'rating']
    u_features = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    u_features = ['user_id']
    for uid, clis in uv_cli_infos[0].items():
        if len(clis) > 1:
            # ???????????????val???
            u_fea_val = clientsdata[clis[0]][clientsdata[clis[0]]["user_id"] == uid].values[0][:5]
            # ??????emb????????????
            for fea, fea_val in zip(u_features, u_fea_val):
                emb_name = "net.embed_layers.embed_" + fea + ".weight"
                if modletype == 'mlp':
                    emb_name = "net.embed_" + fea + ".weight"
                avg_emb, fea_labels = get_agg_emb(nets, clis, fea, fea_val, emb_name, fea_val_label, use_cuda, same_dim,
                                                  my_pca)
                # ??????
                for cli, fea_label in zip(clis, fea_labels):
                    nets[cli].state_dict()[emb_name][fea_label].copy_(avg_emb)


def update_item_emb_avg(nets, clientsdata, uv_cli_infos, fea_val_label, modletype, use_cuda, same_dim):
    # v_features = ['movie_id', 'genre']
    v_features = ['movie_id']

    for vid, clis in uv_cli_infos[1].items():
        if len(clis) > 1:
            # item?????????lable???
            v_fea_values = clientsdata[clis[0]][clientsdata[clis[0]]["movie_id"] == vid].values[0][5:-1]
            # ??????emb????????????
            for fea, fea_val in zip(v_features, v_fea_values):
                emb_name = "net.embed_layers.embed_" + fea + ".weight"
                if modletype == 'mlp':
                    emb_name = "net.embed_" + fea + ".weight"
                avg_emb, fea_labels = get_agg_emb(nets, clis, fea, fea_val, emb_name, fea_val_label, use_cuda, same_dim,
                                                  my_pca)
                # ??????
                for cli, fea_label in zip(clis, fea_labels):
                    nets[cli].state_dict()[emb_name][fea_label].copy_(avg_emb)


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
    # ??????????????????emb div,???????????????lable??????
    fea_val_lable_dict = {}
    # fea_emb_map = {'genre': 216}  # key:valus
    fea_embDim = {}
    for fea in features:
        # ???????????????value???
        fea_value = cli[fea].unique()
        # ???????????????label???
        le = LabelEncoder()
        cli[fea] = le.train_forward_transform(cli[fea])
        fea_label = cli[fea].unique()
        fea_val_lable_dict[fea] = dict(zip(fea_value, fea_label))
        fea_embDim[fea] = cli[fea].nunique()
    return cli, fea_val_lable_dict, fea_embDim


def create_dataloader(df, lossname, modle, use_cuda, batch_size=256):
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

    # ??????Tensor???????????????
    dataset = TensorDataset(tensor_x, tensor_y.view(-1))
    # ??????DataLoader???????????????
    train_data = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    return train_data  # , valid_data, test_data


def get_uv_cli_info(all_data, cli_data):
    # ??????/???????????????
    uid_list = np.sort(all_data['user_id'].unique())
    vid_list = np.sort(all_data['movie_id'].unique())
    fea_uid_map = {}
    fea_vid_map = {}
    uv_data = []
    # ???????????????????????????
    for data in cli_data:
        uv_data.append([data["user_id"].unique(), data["movie_id"].unique()])
    # uid???vid???????????????
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
    # ??????????????????
    import ast
    pa = path + 'u_cli_info.csv'
    u_cli = pd.read_csv(pa, converters={'u_clis': ast.literal_eval})
    fea_uid_map = {}
    for i in range(len(u_cli)):
        fea_uid_map[u_cli.loc[i]['user_id']] = u_cli.loc[i]['u_clis']
    # ???????????????
    pa = path + 'v_cli_info.csv'
    v_cli = pd.read_csv(pa, converters={'v_clis': ast.literal_eval})
    fea_vid_map = {}
    for i in range(len(v_cli)):
        fea_vid_map[v_cli.loc[i]['movie_id']] = v_cli.loc[i]['v_clis']

    return [fea_uid_map, fea_vid_map]


def readdata_ML100k(path, Step, lossnames, modle, use_cuda, is_same_emb=True, istest=True, file=None):
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
    # ??????????????????????????????
    train_datas = []
    # ???????????????????????????????????????
    valid_datas = []
    # ????????????????????????
    total_valid_data = pd.DataFrame()
    # ???????????????????????????????????????
    test_datas = []
    # ????????????????????????
    total_test_data = pd.DataFrame()
    # ????????????????????????????????????
    client_data = []
    # ???????????????????????????
    fea_embInfo = []
    fea_value_label = []
    total_len = 0
    for idx in range(len(files)):
        columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_id', 'genre', 'rating']
        # ??????????????????????????????
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
        # ???????????????????????????????????????

        # train_df = df.sample(frac=0.8, random_state=123, replace=False)
        # other = df[~df.index.isin(train_df.index)]
        # valid_df = other.sample(frac=0.5, random_state=123, replace=False)
        # total_valid_data = total_valid_data.append(valid_df, ignore_index=True)
        # test_df = other[~other.index.isin(valid_df.index)]
        # total_test_data = total_test_data.append(test_df, ignore_index=True)

        test_df = df.sample(frac=0.2, random_state=123, replace=False)
        total_test_data = total_test_data.append(test_df, ignore_index=True)
        other = df[~df.index.isin(test_df.index)]
        train_df = other.sample(frac=0.5, random_state=123, replace=False)
        valid_df = other[~other.index.isin(train_df.index)]
        total_valid_data = total_valid_data.append(valid_df, ignore_index=True)
        # DataLoader??????
        batch_size = int(len(train_df) / Step)
        batch_size = 512
        print("batch_size: ", batch_size)
        train = create_dataloader(train_df, lossname, modle, use_cuda, batch_size)
        train_datas.append(train)
        valid = create_dataloader(valid_df, lossname, modle, use_cuda)
        valid_datas.append(valid)
        test = create_dataloader(test_df, lossname, modle, use_cuda)
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
    # ???????????????????????????????????????
    train = all_data.sample(frac=0.7, random_state=123, replace=False)
    other = all_data[~all_data.index.isin(train.index)]
    valid = other.sample(frac=0.4, random_state=123, replace=False)
    test = other[~other.index.isin(valid.index)]
    # DataLoader??????
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


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params???
        num: int???the number of loss
        x: multi-task loss
    Examples???
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

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


# ???????????????????????????????????????
class DNN(nn.Module):
    def __init__(self, hiddenUnits, dropout=0.3):
        super(DNN, self).__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hiddenUnits[:-1], hiddenUnits[1:]))])
        self.dropout = nn.Dropout(p=dropout)

    # ??????????????? ??????dnn_network??? ???????????????
    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, embInfo, hiddenUnits, lossname, emb_dim):
        super(MLP, self).__init__()
        self.embed_user_id = nn.Embedding(num_embeddings=embInfo['user_id'], embedding_dim=emb_dim)
        self.embed_movie_id = nn.Embedding(num_embeddings=embInfo['movie_id'], embedding_dim=emb_dim)
        """ fully connected layer """
        # self.MLP_Layers = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(Layers[:-1], Layers[1:]))])
        self.dnn_network = DNN(hiddenUnits)
        self.dense_final = nn.Linear(hiddenUnits[-1], 1)
        self.get_auc = get_auc
        # dnn ???
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
        """ ?????? """
        inputs = inputs.long()
        embed_user_id = self.embed_user_id(inputs[:, 0])
        embed_movie_id = self.embed_movie_id(inputs[:, 1])
        """ ?????? """
        embedding_cat = torch.cat((embed_user_id, embed_movie_id), dim=1)
        # """ ?????? """
        # embedding_vec = torch.mul(embed_user_id, embed_movie_id)
        """ ????????? """
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
        # ????????????????????????
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.rand(input_dim, 1))
            for i in range(self.layer_num)
        ])
        self.cross_bias = nn.ParameterList([
            nn.Parameter(torch.rand(input_dim, 1))
            for i in range(self.layer_num)
        ])

    def forward(self, x):
        # x???(None, dim)???????????? ????????????????????????(None, dim, 1)
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
        # embedding??????????????????embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=embDim)
            for key, val in embInfo.items()
        })

        # hiddenUnits.insert(0, len(embInfo) * embDim)

        self.cross_network = CrossDcn(layer_num, hiddenUnits[0])  # layer_num??????????????????????????? hiddenUnits[0]?????????????????????????????????
        self.dnn_network = DNN(hiddenUnits)
        self.dense_final = nn.Linear(hiddenUnits[-1] + hiddenUnits[0], 1)
        self.lossname = lossname

        if lossname == 'auc':
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
        # p??????
        self.w_p = nn.Parameter(torch.rand([sparse_num, sparse_num, hiddenUnits[0]]))  # [26,26,256]
        self.l_b = torch.rand([hiddenUnits[0], ], requires_grad=True)
        if use_cuda:
            self.l_b = self.l_b.cuda()

    def forward(self, z, inputs_embeds):
        # l_z = torch.mm([128,70], [70,64])=[128,64]
        l_z = torch.mm(z.reshape(z.shape[0], -1),
                       self.w_z.permute((2, 0, 1)).reshape(self.w_z.shape[2], -1).T)  # (None, hiddenUnits[0])
        # matmul??????????????????????????????, : [2,5,3]*[1,3,4]->[2,5,4]
        # lp = matmul([128,7,10], [128, 10, 7])=[128, 7, 7]
        p = torch.matmul(inputs_embeds, inputs_embeds.permute((0, 2, 1)))  # [None, sparse_num, sparse_num]
        # mm([128,49],[49,64])=[128,64]
        l_p = torch.mm(p.reshape(p.shape[0], -1),
                       self.w_p.permute((2, 0, 1)).reshape(self.w_p.shape[2], -1).T)  # [None, hiddenUnits[0]]
        # output = [128,64]+[128,64]+[64]=[128,64]
        output = l_z + l_p + self.l_b
        return output


# PNN??????
# ???????????????????????????????????????) -> embedding??? -> crosspnn ??? -> DNN -> ??????
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

        # embedding??????????????????embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=self.embDim)
            for key, val in self.embInfo.items()
        })

        # crosspnn???
        self.crosspnn = CrossPnn(embDim, self.sparse_num, hiddenUnits, use_cuda)

        # dnn ???
        # hiddenUnits[0] += self.dense_num  # dense_inputs???????????????dnn?????????embedding 256+13=269
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
        user_emb = self.embed_layers['embed_user_id'](inputs[:, 0])
        inputs_embeds.append(user_emb)
        movie_emb = self.embed_layers['embed_movie_id'](inputs[:, 1])
        inputs_embeds.append(movie_emb)
        inputs_embeds = torch.stack(inputs_embeds)  # [fea_num, batch_sz, embDim]->[7,128,10]
        # [None, sparse_num, embDim]  ???????????????????????? ????????????????????????view??????reshape
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
                                  self.domain_num, self.mem_num, self.embDim, self.droprate)  # ???1???18???10???
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
        self.embed_user_id = nn.Embedding(num_embeddings=embInfo['user_id'], embedding_dim=embDim)
        self.embed_movie_id = nn.Embedding(num_embeddings=embInfo['movie_id'], embedding_dim=embDim)
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
        # dnn ???
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
        user_emb = self.embed_user_id(inputs[:, 0])
        item_emb = self.embed_movie_id(inputs[:, 1])
        # ???
        if self.cross_type == 'crossmy':
            # share_domain = torch.matmul(self.private_domain, self.shared_memory_d)
            emb_u = torch.unsqueeze(user_emb, dim=2)
            emb_v = torch.unsqueeze(item_emb, dim=2)
            # metaU = torch.matmul(torch.matmul(user_emb, self.shared_memory_d), self.domain_weights_u) + self.domain_bias_u
            # meta_v = torch.matmul(torch.matmul(item_emb, self.shared_memory_d), self.domain_weights_v) + self.domain_bias_v
            metaU = torch.matmul(user_emb, self.shared_memory_d)
            meta_v = torch.matmul(item_emb, self.shared_memory_d)
            metaEmb = torch.cat((metaU, meta_v), 1)
            # metaEmb = F.relu(metaEmb)
            l1 = torch.squeeze(metaEmb)
        elif self.cross_type == 'crossmeta':
            metaV = self.metaItemEmb(item_emb)
            metaEmb = torch.cat((user_emb, metaV), 1)
            l1 = torch.squeeze(metaEmb)
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
                l1 = F.relu(out)  # [128???64]
        # l1 = self.dnn_network(l1)
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


"????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????local train/test??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????"

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


# ??????AUC
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


# -*- coding: utf-8 -*-
import datetime
import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable


# from torchkeras.summary import summary

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

    def fit(self, train_data, backward=True, valid_data=None, epochs=1, log_step_freq=1000):

        # print("Start Training ...")
        # TorchKerasModelCopy.print_bar()
        valid_data = valid_data if valid_data else []

        # for epoch in range(1, epochs + 1):
        # 1???training loop -------------------------------------------------
        train_metrics_sum, step = {}, 0
        loss_epoch = 0
        for features, labels in train_data:
            step = step + 1
            train_metrics, loss = self.train_step(features, labels, backward)
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

        # 2???validate loop -------------------------------------------------

        val_metrics_sum, step = {}, 0
        for features, labels in valid_data:
            step = step + 1
            val_metrics = self.evaluate_step(features, labels)
            for name, metric in val_metrics.items():
                val_metrics_sum[name] = val_metrics_sum.get(name, 0.0) + metric
        for name, metric_sum in val_metrics_sum.items():
            self.history[name] = self.history.get(name, []) + [metric_sum / step]

        # 3???print logs -------------------------------------------------
        infos = {"cli ": self.client_idx}
        infos.update({k: round(self.history[k][-1], 3) for k in self.history})
        tb = PrettyTable()
        tb.field_names = infos.keys()
        tb.add_row(infos.values())
        # print("\n", tb)
        # TorchKerasModelCopy.print_bar()

        # print("Finished Training...")

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


# torchkeras
def init_net(cli_nums, uv_cli_info, use_cuda, same_dim, fea_embInfo, modletype, cross_type, meta=False):
    global net
    # import torchkeras
    emb_dim = 64
    hiddenUnits = [emb_dim * 2, emb_dim * 2, 64, 32, 16]
    if cross_type == 'torchcat': hiddenUnits = [emb_dim, 64, 32, 16]
    local_nets = []
    # ???????????????
    # $from kaggle.MAMLv1 import MetaLearner
    for idx in range(cli_nums):
        lossname = 'auc'
        print(modletype)
        embInfo = fea_embInfo if same_dim else fea_embInfo[idx]
        if modletype == 'mlp':
            # hiddenUnits = [20, 64, 32, 16]
            net = MLP(embInfo, hiddenUnits, lossname, emb_dim)
        elif modletype == 'metamf':
            net = MetaMF(embInfo, lossname)
        elif modletype == 'pnn':
            net = PNN(embInfo, hiddenUnits, use_cuda, lossname, emb_dim)
        elif modletype == 'dcn':
            # hiddenUnits = [20, 64, 32, 16]
            net = DCN(embInfo, hiddenUnits, 3, use_cuda, lossname, embDim=emb_dim)
        elif modletype == 'mynet':
            domain_id = torch.tensor(idx).long()
            if use_cuda: domain_id = domain_id.cuda()
            embInfo['domain_id'] = cli_nums
            # hiddenUnits = [emb_dim, 64, 32, 16]

            net = myNet(embInfo, domain_id, uv_cli_info, hiddenUnits, use_cuda, lossname, cross_type)
        else:
            import warnings
            warnings.warn('modletype argument wrong', UserWarning)

        if use_cuda: net.cuda()
        net.apply(weights_init)

        net = TorchKerasModelCopy(net)
        local_nets.append(net)
    return local_nets


def fed_torchkeras(T, fed_type, train_datas, valid_datas, test_datas, total_valid_datas,
                   total_test_datas, fea_emb_info,
                   uv_cli_info, client_data, fea_val_label, use_cuda, modletype, same_dim, test_type, lossnames,
                   cross_type):
    cli_nums = len(train_datas)
    local_nets = init_net(cli_nums, uv_cli_info, use_cuda, same_dim, fea_emb_info, modletype, cross_type)
    cols_auc = ["epoch", "loss", 'auc', "val_loss", 'val_auc', "fed_loss", 'fed_auc']
    cols_cli = []
    best_net = {}
    for i in range(cli_nums):
        cols_cli.append("train_cli_" + str(i))
        cols_cli.append("fed_cli_" + str(i))
        best_net[i] = [-1, None]
    df_cli = pd.DataFrame(columns=cols_cli)
    df_auc = pd.DataFrame(columns=cols_auc)
    # ????????????
    lr = 0.001
    for t in range(T):
        if (t + 1) % 3 == 0:
            lr = float("%.4f" % (lr * np.power(0.94, 4)))
            # lr = round(lr * np.power(0.97, (T / 3)), 4)
        print("decay_lr: ", lr)
        train_auc = pd.DataFrame()
        for idx in range(cli_nums):
            print("--------------------------------------------------------client",
                  idx + 1, modletype, fed_type, "t", t + 1)
            # ?????? torch.nn.BCELoss()
            loss_func = local_nets[idx].net.lossfunc
            optimizer = torch.optim.Adam(params=local_nets[idx].net.parameters(), lr=lr, weight_decay=0.001)
            metrics_dict = local_nets[idx].net.metricfunc
            cli_idx = idx + 1
            local_nets[idx].compile(loss_func=loss_func, optimizer=optimizer, metrics_dict=metrics_dict, idx=cli_idx)
            # ????????????support,????????????
            met, loss = local_nets[idx].fit(train_data=train_datas[idx], backward=True, valid_data=valid_datas[idx])
            train_auc = pd.concat([train_auc, met], axis=0)
            # ????????????
            # if fed_type == 'Central' and (t + 1) % 1 == 0 and float("%.4f" % met['auc']) > best_net[idx][0]:
            if (t + 1) % 1 == 0 and float("%.4f" % met['auc']) > best_net[idx][0]:
                print("-----------------------------------------------------------save net")
                best_net[idx][0] = float("%.4f" % met['auc'])
                best_net[idx][1] = local_nets[idx]
        # ????????????????????????????????????
        print(train_auc)
        print(train_auc.mean(axis=0))

        # ????????????
        if fed_type == "FedAvg":
            average_all_layers(local_nets)
        elif fed_type == "FedEmb":
            update_user_emb_avg(local_nets, client_data, uv_cli_info, fea_val_label, modletype, use_cuda, same_dim)
            update_item_emb_avg(local_nets, client_data, uv_cli_info, fea_val_label, modletype, use_cuda, same_dim)
        elif fed_type == 'Central':
            continue
        else:
            import warnings
            warnings.warn('fed argument wrong', UserWarning)
        # ????????????

        train_auc = pd.DataFrame()
        # print("\n----------------------------------------------------------------after fed, train on data b")
        # for idx in range(cli_nums):
        #     loss_func = local_nets[idx].net.lossfunc
        #     optimizer = torch.optim.Adam(params=local_nets[idx].net.parameters(), lr=lr, weight_decay=0.001)
        #     metrics_dict = local_nets[idx].net.metricfunc
        #     local_nets[idx].compile(loss_func=loss_func, optimizer=optimizer, metrics_dict=metrics_dict, idx=idx + 1)
        #     met1, loss = local_nets[idx].fit(train_data=valid_datas[idx])
        #     train_auc = pd.concat([train_auc, met1], axis=0)

        # # ????????????
        # if (t + 1) % 1 == 0 and float("%.4f" % met1['auc']) > best_net[idx][0]:
        #     print("-----------------------------------------------------------save net")
        #     best_net[idx][0] = float("%.4f" % met1['auc'])
        #     best_net[idx][1] = local_nets[idx]

        # print(train_auc)
        # print(train_auc.mean(axis=0))

    # ????????????
    print("\ntest model:")
    test_auc = pd.DataFrame()
    for idx in range(cli_nums):
        # ?????????????????????
        met = pd.DataFrame([best_net[idx][1].evaluate(test_datas[idx])])  # ??????????????????
        test_auc = pd.concat([test_auc, met], axis=0)
        # print("cli:%d loss: %5f  auc:%5f" % (idx + 1, met["val_loss"], met.iloc[0][1]))
    # ??????????????????????????????
    print(test_auc)
    print(test_auc.mean(axis=0))
    return test_auc.mean(axis=0), best_net, df_auc, -1, df_cli, cols_cli


def fed_meta(T, fed_type, train_datas, valid_datas, test_datas, total_valid_datas,
             total_test_datas, fea_emb_info,
             uv_cli_info, client_data, fea_val_label, use_cuda, modletype, same_dim, test_type, lossnames,
             cross_type):
    cli_nums = len(train_datas)
    local_nets = init_net(1, uv_cli_info, use_cuda, same_dim, fea_emb_info, modletype, cross_type, meta=False)
    net = local_nets[0]
    for epo in range(100):
        step = 0
        val_loss = 0
        val_auc = 0
        for spt, qry in zip(train_datas[0], valid_datas[0]):
            step += 1
            ## ???0?????????
            optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01, weight_decay=0.001)
            net.train()
            # forward
            pred = net.forward(spt[0])
            loss1 = net.lossfunc(pred, spt[1])
            spt_loss = loss1.item()
            spt_auc = net.get_auc(pred, spt[1]).item()

            # update parameters
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            # ?????????????????????????????????query???????????????
            # net.eval()
            # with torch.no_grad():
            pred = net.forward(qry[0])  # , fast_weights, bn_training = True)
            loss2 = net.lossfunc(pred, qry[1])
            val_auc += net.get_auc(pred, qry[1])
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            val_loss += loss2.item()
            if step % 50 == 0:
                print("loss: ", epo, val_loss / step)

        print("auc: ", val_auc / step)
    return 0  # ,loss


def train_web():
    use_cuda = torch.cuda.is_available()
    use_cuda = False
    print(use_cuda)
    test_type = 'single'
    is_same_emb = True

    path1 = '../dataset/preprocessed_data/ml-100k/non_iid/'
    path2 = '../dataset/preprocessed_data/ml-1m/domain_multiple/'
    path3 = "../input/domainsingle/"
    path4 = "../input/domain-multiple/"

    # ????????????
    files_all = ['Children', 'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
                 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                 'Thriller', 'War', 'Western']

    file_5_big = ['Drama', 'Comedy', 'Action', 'Thriller',
                  'Sci-Fi']  # ['Horror', 'Animation', 'Mystery', 'Musical', 'Fantasy']#
    file_5_small = ['Western', 'Film-Noir', 'Fantasy', 'Musical', 'Mystery']
    # file_10 = ['Drama', 'Comedy', 'Action', 'Thriller', 'Sci-Fi', 'Romance', 'Adventure', 'Crime', 'Children', 'War']
    file_10 = ['Children', 'Horror', 'Animation', 'Mystery', 'Musical', 'Fantasy', 'Romance', 'Adventure', 'Crime',
               'War']
    file_15 = ['Drama', 'Comedy', 'Action', 'Thriller', 'Sci-Fi', 'Romance', 'Adventure', 'Crime', 'Children', 'War',
               'Horror', 'Animation', 'Mystery', 'Musical', 'Fantasy']
    file_2 = ['Film-Noir', 'Western']

    fed_types = ["Central", "FedAvg", "FedEmb", "FedDomainCat", "FedDomainCross"]
    fed_type = "FedAvg"
    modles = ['mlp', 'pnn', 'dcn', 'metamf', 'mynet']
    modle = 'mlp'
    lossname = 'auc'
    Step = 200
    # ?????????
    path = path4
    Epoch = 100
    file_list = [file_15]  # ,file_10, file_15]#file_10,
    is_test = False  # False True
    auc_log = {}
    for modle in modles[:3]:
        for fed_type in fed_types[2:3]:
            for file in file_list:
                print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                train_datas, valid_datas, test_datas, total_valid_datas, total_test_datas, fea_emb_dims, uv_cli_info, client_data, fea_value_label = readdata_ML100k(
                    path, Step, lossname, modle, use_cuda, is_same_emb=is_same_emb, istest=is_test, file=file)
                # ??????
                test_auc, best_net, df_avg, df_cli, cols_cli, _ = fed_torchkeras(Epoch, fed_type, train_datas,
                                                                                 valid_datas, test_datas,
                                                                                 total_valid_datas, total_test_datas,
                                                                                 fea_emb_dims, uv_cli_info, client_data,
                                                                                 fea_value_label,
                                                                                 use_cuda, modle, same_dim=is_same_emb,
                                                                                 test_type=test_type,
                                                                                 lossnames=lossname, cross_type="none")
                name = modle + fed_type
                auc_log[name] = (test_auc)
        print('\n ----------------------------------------------------------------------------------------auc_log \n')
        for name, val in auc_log.items():
            print(name, ":", val, '\n')
        # _ = fed_meta(Epoch, fed_type, train_datas, valid_datas, test_datas,
        #                                                 total_valid_datas, total_test_datas,
        #                                                 fea_emb_dims, uv_cli_info, client_data, fea_value_label,
        #                                                 use_cuda,modle, same_dim=is_same_emb, test_type=test_type,
        #                                                 lossnames=lossname, cross_type="none")


train_web()

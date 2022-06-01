import torch
import pandas as pd
import numpy as np
from kaggle.kg import create_dataloader, weights_init, PNN, get_mae, get_mse, get_rmse


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


def train_central():
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    path = '../dataset/preprocessed_data/ml-100k/non_iid/'
    is_test = False
    modletype = 'pnn'
    lossname = 'mae'
    T = 2
    # 读取数据
    train_datas, valid_datas, test_datas, fed_emb_infos = readdata_ML100k_cnetral(path, lossname, modletype, use_cuda, is_test)
    hidden_units = [64, 32, 16]
    net = PNN(fed_emb_infos, hidden_units, use_cuda)
    if use_cuda:
        net.cuda()
    import torchkeras
    net = torchkeras.Model(net)
    print(net)
    net.apply(weights_init)
    lr = 0.001
    for t in range(T):
        if (t + 1) % 15 == 0:
            lr = round(lr * np.power(0.97, (T / 10)), 4)
        print("decay_lr: ", lr)
        # 参数
        net.compile(loss_func=torch.nn.MSELoss(),
                                optimizer=torch.optim.Adam(params=net.parameters(), lr=lr,
                                                           weight_decay=0.001),
                                metrics_dict={"mae": get_mae, "mse": get_mse, "rmse": get_rmse})
        # 训练
        net.fit(epochs=1, dl_train=train_datas, dl_val=valid_datas,
                                  log_step_freq=3000)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# train_central()

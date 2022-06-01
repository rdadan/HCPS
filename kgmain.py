# from kaggle.kg import *
# from kaggle.kg_awl import *
from kaggle.kg_meta import *
from kaggle.MAMLv1 import *

def train_onpc():
    use_cuda = torch.cuda.is_available()
    use_cuda = False
    is_test = True

    print(use_cuda)
    test_type = 'single'
    is_same_emb = True

    path1 = '../dataset/preprocessed_data/ml-100k/non_iid/'
    path2 = '../dataset/preprocessed_data/ml-1m/domain_multiple/'
    path3 = "../input/domainsingle/"
    path4 = "../input/domain-multiple/"
    # 读取数据

    files = ['Children', 'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
             'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
             'Thriller', 'War', 'Western']
    file_5 = ['Drama', 'Comedy', 'Action', 'Thriller', 'Sci-Fi']
    file_10 = ['Drama', 'Comedy', 'Action', 'Thriller', 'Sci-Fi', 'Romance', 'Adventure', 'Crime', 'Children', 'War']
    file_15 = ['Drama', 'Comedy', 'Action', 'Thriller', 'Sci-Fi', 'Romance', 'Adventure', 'Crime', 'Children', 'War',
               'Horror', 'Animation', 'Mystery', 'Musical', 'Fantasy']
    file_2 = ['Film-Noir', 'Western']
    path = path2
    files = file_2
    lossname = 'auc'
    cross_type = 'crossmeta'#crossdcn' crossmy
    fed_avg = "FedAvg"
    fed_emb = "FedEmb"
    fed_central = "Central"
    fed_domainCross = "FedDomainCross"
    fed_domainCat = "FedDomainCat"
    fed_type = fed_central
    Epoch =3
    train_step = 100
    valid_step = 5
    modle = 'dcn'#mynet2, 'metamf'#'mynet'

    train_datas, valid_datas, test_datas, total_valid_datas, total_test_datas, fea_emb_dims, uv_cli_info, client_data, fea_value_label = \
        readdata_ML100k(
        path, train_step, valid_step, lossname, modle, use_cuda, is_same_emb=is_same_emb, istest=is_test, file=files)

    # best_net, df_auc, df_mae, df_cli, cols_cli = fed_torchkeras(Epoch, train_step, valid_step, fed_type, train_datas, valid_datas, test_datas,
    #                                                     total_valid_datas, total_test_datas,
    #                                                     fea_emb_dims, uv_cli_info, client_data, fea_value_label,
    #                                                     use_cuda,modle, same_dim=is_same_emb, test_type=test_type,
    #                                                     lossnames=lossname, cross_type=cross_type)



    print("save net done xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")


#
train_onpc()



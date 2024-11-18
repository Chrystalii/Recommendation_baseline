import os
import torch
import numpy as np
from scipy import sparse
from datetime import datetime

def get_cur_time():
    # 获取当前时间
    now = datetime.now()
    # 定义时间的返回格式
    cur_time = now.strftime('%Y-%m-%d-%H-%M-%S')
    return cur_time

def create_res_dir(args):
    if len(args.data_dir) == 0:
        args.data_dir = os.path.join(args.project_dir, "data", args.dataset, "output")
        if args.dataset == "taobao":
            args.data_dir += "_res"
    if args.dataset == "netflix":
        args.dataset = "nf"
    cur_time = get_cur_time()
    res_file_name = "{}-{}_{}".format(args.dataset, args.model_name, cur_time)
    args.res_dir = os.path.join(args.data_dir, res_file_name)
    logs_dir = os.path.join(args.res_dir, "log")
    model_dir = os.path.join(args.res_dir, "ckpt")
    test_log_dir = os.path.join(args.res_dir, "test_log")

    if not os.path.exists(args.res_dir):
        os.makedirs(logs_dir)
        os.makedirs(model_dir)
        os.makedirs(test_log_dir)

    logs_file = os.path.join(logs_dir, "{}_{}.log".format(args.dataset, args.model_name))
    args.log = logs_file

def read_datasets(data_dir):
    train_file_path = os.path.join(data_dir, "train_coo_record.npz")
    train_skew_file_path = os.path.join(data_dir, "train_skew_coo_record.npz")
    val_file_path = os.path.join(data_dir, "val_coo_record.npz")
    test_file_path = os.path.join(data_dir, "test_coo_record.npz")

    train_data, num_users, num_items = read_file(train_file_path, True)
    train_skew_data = read_file(train_skew_file_path)

    train = np.vstack((train_data, train_skew_data))
    val = read_file(val_file_path)
    test = read_file(test_file_path)

    return num_users, num_items, train, val, test

def read_file(path, flag=False):
    data = np.load(path)

    users, items, preference = data['row'], data['col'], data['data']
    dataset = np.vstack((users, items, preference)).transpose()

    if flag:
        num_users, num_items = data['shape']
        return dataset, num_users, num_items
    return dataset

def compute_matrics(X_test, X, k=20):
    recall = compute_recall(X_test, X, k)
    hit_ratio = compute_hit_ratio(X_test, X, k)
    ndcg = compute_ndcg(X_test, X, k)
    return recall, hit_ratio, ndcg

def compute_recall(outputs, labels, k=20):
    _, preds = torch.topk(outputs, k, sorted=False)  # top k index
    rows = torch.arange(labels.shape[0]).view(-1, 1)

    recall = torch.sum(labels[rows, preds], dim=1) \
             / torch.min(torch.Tensor([k]), torch.sum(labels, dim=1))
    recall[torch.isnan(recall)] = 0
    return recall

def compute_hit_ratio(outputs, labels, k=20):
    _, preds = torch.topk(outputs, k, sorted=False)  # top k index
    rows = torch.arange(labels.shape[0]).view(-1, 1)

    # Check if any of the top-k predictions match the true labels
    hit_ratio = torch.sum(labels[rows, preds], dim=1).clamp(max=1)
    return hit_ratio

def compute_ndcg(outputs, labels, k=20):
    _, preds = torch.topk(outputs, k)  # sorted top k index of outputs
    _, facts = torch.topk(labels, k)  # min(k, labels.nnz(dim=1))
    rows = torch.arange(labels.shape[0]).view(-1, 1)

    tp = 1.0 / torch.log2(torch.arange(2, k + 2).float())
    dcg = torch.sum(tp * labels[rows, preds], dim=1)
    idcg = torch.sum(tp * labels[rows, facts], dim=1)
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0
    return ndcg

def change_csr_matrix(data, n_users, n_items):
    users, items = data[:, 0], data[:, 1]
    res = sparse.csr_matrix((np.ones_like(users), (users, items)),
                            shape=(n_users, n_items), dtype=np.float)
    return res

def change_neg_info(data, n_users, n_items):
    csr_data = change_csr_matrix(data, n_users, n_items)
    all_pos = []
    for i in range(n_users):
        item_info = csr_data[i].indices
        all_pos.append(list(item_info))

    for i in range(len(data)):
        neg_item = np.random.randint(n_items)
        while neg_item in all_pos[int(data[i][0])]:
            neg_item = np.random.randint(n_items)
        data[i][2] = neg_item
    return data, csr_data

def save_results(args, results):
    res_path = os.path.join(args.res_dir, "test_log",
                            "{}_{}_test.log".format(args.dataset, args.model_name))
    with open(res_path, 'w', encoding='utf-8') as f:
        f.write("best epoch: {}\n".format(args.best_epoch))
        for key, value in results.items():
            f.write("TEST results {}\n".format(key))
            f.write("recall: {}\n".format(value[0]))
            f.write("hit_ratio: {}\n".format(value[1]))
            f.write("ndcg: {}\n".format(value[2]))
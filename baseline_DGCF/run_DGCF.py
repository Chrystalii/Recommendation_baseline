import numpy as np
import torch
import torch.optim as optim
import logging
import sys
import time
import argparse
from torch.utils.data import DataLoader

from DGCF import DGCF
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Run DGCF.")
    parser.add_argument('--data_dir', nargs='?', default='',
                        help='Input data path.')
    parser.add_argument('--project_dir', nargs='?', default='D:/Code/Python/RecommendationSystem/DICE-main',
                        help='Project path.')
    parser.add_argument('--model_name', default='DGCF', type=str, help='model name')
    parser.add_argument('--pick', type=int, default=0,
                        help='O for no pick, 1 for pick')
    parser.add_argument('--pick_scale', type=float, default=1e10,
                        help='Scale')
    parser.add_argument('--dataset', nargs='?', default='netflix',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book, ml10m, netflix, taobao}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1:Use stored models.')
    parser.add_argument('--embed_name', nargs='?', default='',
                        help='Name for pretrained model.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')

    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epochs')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=2000,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--cor_flag', type=int, default=1,
                        help='Correlation matrix flag')
    parser.add_argument('--corDecay', type=float, default=0.01,
                        help='Distance Correlation Weight')
    parser.add_argument('--regs', nargs='?', default='[1e-3,1e-4,1e-4]',
                        help='Regularizations.')

    parser.add_argument('--n_layers', type=int, default=1,
                        help='Layer numbers.')
    parser.add_argument('--n_factors', type=int, default=4,
                        help='Number of factors to disentangle the original embed-size representation.')
    parser.add_argument('--n_iterations', type=int, default=2,
                        help='Number of iterations to perform the routing mechanism.')

    parser.add_argument('--show_step', type=int, default=3,
                        help='Test every show_step epochs.')
    parser.add_argument('--early', type=int, default=40,
                        help='Step for stopping')
    parser.add_argument('--Ks', nargs='?', default='[20, 50]',
                        help='Metrics scale')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Save Better Model')
    parser.add_argument('--save_name', nargs='?', default='best_model',
                        help='Save_name.')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--res_dir', nargs='?', default='../test',
                        help='Path to save res and log')

    return parser.parse_args()

def print(*args, **kwargs):
    logging.info(' '.join([str(arg) for arg in args]))

# def sample_cor_samples(n_users, n_items, cor_batch_size):
#     r"""This is a function that sample item ids and user ids.
#     Args:
#         n_users (int): number of users in total
#         n_items (int): number of items in total
#         cor_batch_size (int): number of id to sample
#     Returns:
#         list: cor_users, cor_items. The result sampled ids with both as cor_batch_size long.
#
#     Note:
#         We have to sample some embedded representations out of all nodes.
#         Because we have no way to store cor-distance for each pair.
#     """
#     cor_users = random.sample(list(range(n_users)), cor_batch_size)
#     cor_items = random.sample(list(range(n_items)), cor_batch_size)
#
#     return cor_users, cor_items

def data_process(data, num_users):
    temp = data[:, [0, 1]]
    temp[:, 1] += num_users

    # sort
    indices = np.lexsort((temp[:, 1], temp[:, 0]))
    interations = temp[indices]

    interations_hat = interations[:, [1, 0]]
    res_data = np.vstack((interations, interations_hat))

    all_h_list = res_data[:, 0]
    all_t_list = res_data[:, 1]
    all_v_list = np.ones_like(all_h_list)

    return all_h_list, all_t_list, all_v_list

if __name__ == '__main__':
    args = parse_args()
    utils.create_res_dir(args)
    args.device = device
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    Ks = eval(args.Ks)

    print("************************* Run with following settings 🏃 ***************************")
    print(args)
    print("************************************************************************************")

    """
        load datasets
    """
    print("Load dataset")
    num_users, num_items, train_inter, val_inter, test_inter = utils.read_datasets(args.data_dir)
    new_train_inter = np.vstack((train_inter, val_inter))

    print(f"n_users = {num_users}, n_items = {num_items}")
    print(f"n_interactions = {len(new_train_inter) + len(test_inter)}, n_train = {len(new_train_inter)}, n_test = {len(test_inter)}")

    train_data, train_csr = utils.change_neg_info(new_train_inter, num_users, num_items)
    test_csr = utils.change_csr_matrix(test_inter, num_users, num_items)
    """
        Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    all_h_list, all_t_list, all_v_list = data_process(train_data, num_users)

    train_data = torch.tensor(train_data, dtype=torch.int64).to(device)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    cor_users_all = torch.randint(0, num_users, (len(train_data),))
    cor_items_all = torch.randint(0, num_items, (len(train_data),))

    cor_tensor = torch.vstack((cor_users_all, cor_items_all)).T.to(device)
    cor_loader = DataLoader(cor_tensor, batch_size=args.batch_size, shuffle=True)

    config = dict()
    config['n_users'] = num_users
    config['n_items'] = num_items
    config['all_h_list'] = all_h_list
    config['all_t_list'] = all_t_list

    print("Initialize model")
    model = DGCF(args, config)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    """
    Train
    """
    best_res = None
    # for early stopping
    es = 0

    t0 = time.time()
    print("Start training")
    for epoch in range(args.epoch):
        t1 = time.time()
        # Initialize the total loss for this epoch as a tensor on the device
        epoch_loss , mf_loss, emb_loss, cor_loss = 0.0, 0.0, 0.0, 0.0
        for batch_id, (batch_data, batch_cor_data) in enumerate(zip(train_loader, cor_loader)):
            users, pos_items, neg_items = batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]
            cor_users, cor_items = batch_cor_data[:, 0], batch_cor_data[:, 1]
            optimizer.zero_grad()
            # Forward pass and loss computation
            batch_mf_loss, batch_emb_loss, batch_cor_loss, batch_loss = model(users, pos_items, neg_items, cor_users, cor_items)

            # Check for NaN values in the loss
            if torch.isnan(batch_loss):
                print('ERROR: Loss is NaN. Exiting...')
                sys.exit()

            # Accumulate the losses
            epoch_loss = epoch_loss + batch_loss.item()
            mf_loss += batch_mf_loss.item()
            emb_loss += batch_emb_loss.item()
            cor_loss += batch_cor_loss.item()

            batch_loss.backward()
            # Perform optimization step after all batches
            optimizer.step()

        t2 = time.time()
        # Logging the losses for the current epoch
        print(f"Epoch {epoch}, train_cost {(t2 - t1):.5f}s, loss [{epoch_loss:.5f} = {mf_loss:.5f} + {emb_loss:.5f} + {cor_loss:.5f}]")

        # Testing the model every `args.show_step` epochs
        if (epoch + 1) % args.show_step == 0:
            cur_res = {}
            for topK in Ks:
                recall, hit_ratio, ndcg = model.test_model(args, train_csr, test_csr, topK)
                key = "topk = {}".format(topK)
                value = [recall.item(), hit_ratio.item(), ndcg.item()]
                cur_res[key] = value
            t3 = time.time()
            print(f"test_cost {t3 - t2:.5f}s {cur_res}")

            # Early stopping
            # Logging the best test results
            if best_res is None:
                best_res = cur_res
            else:
                if best_res["topk = 20"][0] < cur_res["topk = 20"][0]:
                    best_res = cur_res
                    es = 0
                    args.best_epoch = epoch
                else:
                    es += 1
                    if es > 10:
                        print("Early stopping")
                        break

    # save the best result
    utils.save_results(args, best_res)
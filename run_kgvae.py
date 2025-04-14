#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KGVAE – 結合 RecVAE 與 KGCL 的新模型
1. 利用用戶-項目交互資料建立 user-item 矩陣，作為 RecVAE 的輸入；
2. 利用 RecVAE 得到的使用者潛在表示初始化 KGCL 中使用者向量；
3. 交替訓練 RecVAE（重建用戶交互）以及 KGCL（CF 與 KG損失）。
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from collections import defaultdict
from tqdm import tqdm
from prettytable import PrettyTable
import datetime

# 載入我們現有的模組與工具包
from utils.parser import parse_args_kgcl
from utils.data_loader_kgcl import load_data, generate_kg_batch
from modules.KGCL.KGCL import KGCL
from modules.RecVAE.model import VAE  # 這邊 VAE 用來做 RecVAE
from utils.evaluator import Evaluator
from utils.helper import early_stopping, init_logger
from utils.sampler import UniformSampler

# ---------------------------
# 輔助函數：將 train_cf 轉換為 user-item 交互矩陣
def build_user_item_matrix(train_cf, n_users, n_items, device):
    """
    將 train_cf (shape: [N, 2]) 轉成 shape: (n_users, n_items)
    """
    mat = np.zeros((n_users, n_items), dtype=np.float32)
    for u, i in train_cf:
        mat[int(u), int(i)] = 1.0
    return torch.tensor(mat, device=device)

# ---------------------------
# 定義新模型 KGVAE：包含 RecVAE 與 KGCL
class KGVAE(nn.Module):
    def __init__(self, data_config, args_config, graph, kg_dict, adj_mat, user_item_matrix):
        """
        data_config: 包含 n_users, n_items, n_entities, 等參數的字典
        args_config: 參數
        graph, kg_dict, adj_mat: KGCL 所需的結構資料
        user_item_matrix: 使用者-項目交互矩陣 (shape: n_users x n_items)
        """
        super(KGVAE, self).__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.args = args_config
        self.user_item_matrix = user_item_matrix  # RecVAE 輸入

        # 建立 RecVAE 模組，這裡隱藏層、潛在維度均用 args.dim，確保輸出與 KGCL 使用者向量維度一致
        self.recvae = VAE(hidden_dim=args_config.dim, latent_dim=args_config.dim, input_dim=self.n_items)
        # 建立 KGCL 模組
        self.kgcl = KGCL(data_config, args_config, graph, adj_mat)
        # 初始時使用 RecVAE 的表示初始化 KGCL 使用者部分
        self.update_user_embeddings_from_recvae()

    def update_user_embeddings_from_recvae(self):
        """
        利用 RecVAE 的 encoder（不啟動 dropout）得到的均值更新 KGCL 中使用者部分的向量。
        """
        self.recvae.eval()
        with torch.no_grad():
            mu, _ = self.recvae.encoder(self.user_item_matrix, dropout_rate=0)
        # KGCL 中 all_embed 的前 n_users 行存放使用者向量
        self.kgcl.all_embed.data[:self.n_users, :] = mu

    def forward(self, batch):
        # KGCL 的 forward 實現，CF 及 contrastive loss 均在 KGCL 內部完成
        return self.kgcl(batch)

    def recvae_forward(self, user_ratings, beta=None, gamma=1, dropout_rate=0.5):
        # RecVAE 的 forward，計算重建損失
        return self.recvae(user_ratings, beta=beta, gamma=gamma, dropout_rate=dropout_rate, calculate_loss=True)

    def generate(self):
        return self.kgcl.generate()

# ---------------------------
# 主訓練流程
if __name__ == '__main__':
    # 固定隨機種子
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from setproctitle import setproctitle
    setproctitle('EXP@KGVAE')
    sampling = UniformSampler(seed)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    try:
        # 讀取參數與初始化 logger
        args = parse_args_kgcl()
        device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
        log_fn = init_logger(args)
        from logging import getLogger
        logger = getLogger()
        logger.info("PID: %d", os.getpid())
        logger.info("Experiment Description: %s", args.desc)

        # 讀取資料 (train_cf, test_cf, user_dict, n_params, graph, kg_dict, adj_mat)
        train_cf, test_cf, user_dict, n_params, graph, kg_dict, adj_mat = load_data(args)
        n_users = n_params['n_users']
        n_items = n_params['n_items']

        # 利用 train_cf 建立 user-item 矩陣 (RecVAE 輸入)，shape = (n_users, n_items)
        user_item_matrix = build_user_item_matrix(train_cf, n_users, n_items, device)

        # 建立 KGVAE 模型
        model = KGVAE(n_params, args, graph, kg_dict, adj_mat, user_item_matrix).to(device)
        model.kgcl.print_shapes()

        # 定義 optimizer：分別更新 RecVAE 與 KGCL 部分
        optimizer_recvae = torch.optim.Adam(model.recvae.parameters(), lr=args.lr)
        optimizer_kg = torch.optim.Adam(model.kgcl.parameters(), lr=args.lr)

        evaluator = Evaluator(args)
        test_interval = 1
        early_stop_step = 10
        cur_best_pre_0 = 0
        cur_stopping_step = 0

        logger.info("Start training ...")
        for epoch in range(args.epoch):
            model.train()
            # ----- Stage 1: 訓練 RecVAE 部分 -----
            (mll, kld), recvae_loss = model.recvae_forward(user_item_matrix, beta=None, gamma=1, dropout_rate=0.5)
            optimizer_recvae.zero_grad()
            recvae_loss.backward()
            optimizer_recvae.step()
            # 更新 KGCL 使用者向量
            model.update_user_embeddings_from_recvae()
            recvae_loss_val = recvae_loss.item()

            # ----- Stage 2: KGCL CF 部分訓練 -----
            # 定義負樣本採樣（與原來相同）
            def neg_sampling(train_cf_pairs, train_user_dict):
                t1 = time()
                train_cf_negs = sampling.sample_negative(train_cf_pairs[:, 0], n_items, train_user_dict, 1)
                train_cf_triples = np.concatenate([train_cf_pairs, train_cf_negs], axis=1)
                t2 = time()
                logger.info("Negative sampling time: %.2fs", t2 - t1)
                logger.info("train_cf_triples shape: %s", str(train_cf_triples.shape))
                return train_cf_triples

            train_cf_with_neg = neg_sampling(train_cf, user_dict['train_user_set'])
            indices = np.arange(len(train_cf))
            np.random.shuffle(indices)
            train_cf_with_neg = train_cf_with_neg[indices]

            add_loss_dict = defaultdict(float)
            s = 0
            train_start = time()
            batch_num = len(train_cf) // args.batch_size
            for _ in tqdm(range(batch_num), desc=f"Epoch {epoch} CF Training"):
                def get_feed_dict(cf_data, start, end):
                    feed = {}
                    pairs = torch.from_numpy(cf_data[start:end]).to(device).long()
                    feed['users'] = pairs[:, 0]
                    feed['pos_items'] = pairs[:, 1]
                    feed['neg_items'] = pairs[:, 2]
                    feed['batch_start'] = start
                    return feed
                batch = get_feed_dict(train_cf_with_neg, s, s + args.batch_size)
                # KGCL 內部產生 augmentation views
                batch['aug_views'] = model.kgcl.get_aug_views()
                batch_loss, batch_loss_dict = model(batch)
                optimizer_kg.zero_grad()
                batch_loss.backward()
                optimizer_kg.step()
                for k, v in batch_loss_dict.items():
                    add_loss_dict[k] += v / len(train_cf)
                s += args.batch_size
            cf_time = time() - train_start

            # ----- Stage 3: KGCL KG 部分訓練 -----
            kg_start = time()
            kg_total_loss = 0
            n_kg_batch = n_params['n_triplets'] // 4096
            from utils.data_loader_kgcl import generate_kg_batch
            for _ in tqdm(range(1, n_kg_batch + 1), desc=f"Epoch {epoch} KG Training"):
                kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = generate_kg_batch(
                    kg_dict, 4096, n_params['n_entities']
                )
                kg_batch_head = kg_batch_head.to(device)
                kg_batch_relation = kg_batch_relation.to(device)
                kg_batch_pos_tail = kg_batch_pos_tail.to(device)
                kg_batch_neg_tail = kg_batch_neg_tail.to(device)
                kg_loss = model.kgcl.calc_kg_loss_transE(kg_batch_head, kg_batch_relation,
                                                         kg_batch_pos_tail, kg_batch_neg_tail)
                optimizer_kg.zero_grad()
                kg_loss.backward()
                optimizer_kg.step()
                kg_total_loss += kg_loss.item()
            kg_time = time() - kg_start

            logger.info("Epoch %04d | RecVAE Loss: %.4f | CF Time: %.1fs | KG Time: %.1fs | Mean KG Loss: %.4f",
                        epoch, recvae_loss_val, cf_time, kg_time, kg_total_loss / n_kg_batch)

            # ----- 評估 -----
            if epoch % test_interval == 0 and epoch >= 0:
                model.eval()
                test_start = time()
                with torch.no_grad():
                    ret = evaluator.test(model.kgcl, user_dict, n_params)
                test_time = time() - test_start
                results = PrettyTable()
                results.field_names = ["Epoch", "RecVAE Loss", "CF Time", "KG Time", "Recall", "NDCG", "Precision", "Hit Ratio"]
                results.add_row([epoch, f"{recvae_loss_val:.4f}",
                                 f"{cf_time:.1f}", f"{kg_time:.1f}",
                                 ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']])
                logger.info("\n" + str(results))

                cur_best_pre_0, cur_stopping_step, should_stop = early_stopping(ret['recall'][0],
                                                                                 cur_best_pre_0,
                                                                                 cur_stopping_step,
                                                                                 expected_order='acc',
                                                                                 flag_step=early_stop_step)
                if cur_stopping_step == 0:
                    logger.info("### Found better model at epoch %d", epoch)
                elif should_stop:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

        logger.info("Training complete. Early stopping at epoch %d, recall@20: %.4f", epoch, cur_best_pre_0)

    except Exception as e:
        logger.exception(e)

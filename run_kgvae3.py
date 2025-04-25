#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KGVAE – 結合 RecVAE 與 KGCL 的新模型

訓練流程：
1. 利用用戶-項目交互資料建立 user-item 矩陣，作為 RecVAE 輸入，
   預先訓練 RecVAE（重建用戶交互），直到收斂或達到預設 epoch。
2. 利用預訓練好的 RecVAE 得到的使用者潛在表示，更新 KGCL 中的使用者向量。
3. 固定（或忽略） RecVAE，僅進行 KGCL 部分的訓練（包含 CF 與 KG 損失）。
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
from logging import getLogger
import datetime

# 載入我們現有的模組與工具包
from utils.parser import parse_args_kgcl
from utils.data_loader_kgcl import load_data, generate_kg_batch
from modules.KGCL.KGCL import KGCL
from modules.RecVAE.model import VAE  # 使用 RecVAE 模組
from utils.evaluator import Evaluator
from utils.helper import early_stopping, init_logger
from utils.sampler import UniformSampler


# ---------------------------
# 輔助函數：將 train_cf 轉換為 user-item 交互矩陣
def build_user_item_matrix(train_cf, n_users, n_items, device):
    """
    將 train_cf (shape: [N, 2]) 轉換成 shape: (n_users, n_items)
    每列代表一個使用者對所有項目的交互（此處以 0/1 編碼）
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
        data_config: 包含 n_users, n_items, n_entities 等參數的字典
        args_config: 參數
        graph, kg_dict, adj_mat: KGCL 所需的結構資料
        user_item_matrix: 使用者-項目交互矩陣 (shape: n_users x n_items)
        """
        super(KGVAE, self).__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.args = args_config
        self.user_item_matrix = user_item_matrix  # RecVAE 輸入

        # 建立 RecVAE 模組，隱藏層、潛在維度均用 args.dim，確保輸出與 KGCL 使用者向量維度一致
        self.recvae = VAE(hidden_dim=args_config.dim, latent_dim=args_config.dim, input_dim=self.n_items)
        # 建立 KGCL 模組
        self.kgcl = KGCL(data_config, args_config, graph, adj_mat)

    def update_user_embeddings_from_recvae(self):
        """
        利用預訓練好的 RecVAE 的 encoder（不啟動 dropout）得到的均值，
        更新 KGCL 中的使用者部分的向量（all_embed 的前 n_users 行）。
        """
        self.recvae.eval()
        with torch.no_grad():
            mu, _ = self.recvae.encoder(self.user_item_matrix, dropout_rate=0)
        self.kgcl.all_embed.data[:self.n_users, :] = mu

    def forward(self, batch):
        # 直接呼叫 KGCL 的 forward 進行 CF 與 contrastive loss 的計算
        return self.kgcl(batch)

    def recvae_forward(self, user_ratings, beta=None, gamma=1, dropout_rate=0.5):
        # RecVAE 部分的 forward，計算重建損失
        return self.recvae(user_ratings, beta=beta, gamma=gamma, dropout_rate=dropout_rate, calculate_loss=True)

    def generate(self):
        return self.kgcl.generate()
 # 定義負樣本採樣函數（與原流程相同）
def neg_sampling(cf_pairs, train_user_dict):
    t1 = time()
    cf_negs = sampling.sample_negative(cf_pairs[:, 0], n_items, train_user_dict, 1)
    cf_triples = np.concatenate([cf_pairs, cf_negs], axis=1)
    t2 = time()
    logger.info("Negative sampling time: %.2fs", t2 - t1)
    logger.info("train_cf_triples shape: %s", str(cf_triples.shape))
    return cf_triples

def get_feed_dict(data, start, end):
            feed = {}
            pairs = torch.from_numpy(data[start:end]).to(device).long()
            feed['users'] = pairs[:, 0]
            feed['pos_items'] = pairs[:, 1]
            feed['neg_items'] = pairs[:, 2]
            feed['batch_start'] = start
            return feed
# ---------------------------
# 主訓練流程：分為預訓練 RecVAE 和 KGCL 兩階段
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


        logger = getLogger()
        logger.info("PID: %d", os.getpid())
        logger.info("Experiment Description: %s", args.desc)

        # 讀取資料
        train_cf, test_cf, user_dict, n_params, graph, kg_dict, adj_mat = load_data(args)
        n_users = n_params['n_users']
        n_items = n_params['n_items']

        # 建立 user-item 交互矩陣 (用於 RecVAE 輸入)
        user_item_matrix = build_user_item_matrix(train_cf, n_users, n_items, device)

        # 建立 KGVAE 模型
        model = KGVAE(n_params, args, graph, kg_dict, adj_mat, user_item_matrix).to(device)
        model.kgcl.print_shapes()

        # ─────────────────────────────────────
        # 第一階段：預先訓練 RecVAE 部分
        pretrain_epochs = 50  # 依據實驗需求設置預訓練 epoch 數
        optimizer_recvae = torch.optim.Adam(model.recvae.parameters(), lr=args.lr)
        logger.info("Pretraining RecVAE for %d epochs...", pretrain_epochs)
        for epoch in range(pretrain_epochs):
            model.recvae.train()
            optimizer_recvae.zero_grad()
            # 此處使用全量 user_item_matrix 作為 RecVAE 輸入
            (mll, kld), recvae_loss = model.recvae_forward(user_item_matrix, beta=None, gamma=1, dropout_rate=0.5)
            recvae_loss.backward()
            optimizer_recvae.step()
            logger.info("RecVAE Epoch %d: Loss = %.4f", epoch, recvae_loss.item())
        # 預訓練完成，利用 recvae 更新 KGCL 中的使用者向量
        model.update_user_embeddings_from_recvae()
        logger.info("RecVAE pretraining complete. Updated KGCL user embeddings.")
        # ─────────────────────────────────────
        # 第二階段：僅訓練 KGCL 部分（包括 CF 與 KG 損失），不再更新 RecVAE
        optimizer_kg = torch.optim.Adam(model.kgcl.parameters(), lr=args.lr)
        evaluator = Evaluator(args)
        test_interval = 1
        early_stop_step = 10
        cur_best_pre_0 = 0
        cur_stopping_step = 0

        logger.info("Starting KGCL training...")
        for epoch in range(args.epoch):



            # ----- CF 部分訓練 -----
            # ----- cf data ----- """
            train_cf_with_neg = neg_sampling(train_cf, user_dict['train_user_set'])
            indices = np.arange(len(train_cf))
            np.random.shuffle(indices)
            train_cf_with_neg = train_cf_with_neg[indices]

            # ----- training cf-----"""
            model.kgcl.train()
            aug_views = model.kgcl.get_aug_views()
            add_loss_dict = defaultdict(float)
            s = 0
            train_start = time()
            batch_num = len(train_cf) // args.batch_size
            for _ in tqdm(range(batch_num), desc=f"Epoch {epoch} CF Training"):

                batch = get_feed_dict(train_cf_with_neg, s, s + args.batch_size)
                # KGCL 內部產生 augmentation views
                batch['aug_views'] = aug_views
                batch_loss, batch_loss_dict = model(batch)
                optimizer_kg.zero_grad()
                batch_loss.backward()
                optimizer_kg.step()
                for k, v in batch_loss_dict.items():
                    add_loss_dict[k] += v / len(train_cf)
                s += args.batch_size
            cf_time = time() - train_start

            # ----- KG 部分訓練 -----
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

            logger.info("Epoch %04d | CF Time: %.1fs | KG Time: %.1fs | Mean KG Loss: %.4f",
                        epoch, cf_time, kg_time, kg_total_loss / n_kg_batch)

            # ----- 評估 -----
            if epoch % test_interval == 0 and epoch >= 0:
                model.kgcl.eval()
                test_start = time()
                with torch.no_grad():
                    ret = evaluator.test(model.kgcl, user_dict, n_params)
                test_time = time() - test_start
                results = PrettyTable()
                results.field_names = ["Epoch", "CF Time", "KG Time", "Recall", "NDCG", "Precision", "Hit Ratio"]
                results.add_row([epoch, f"{cf_time:.1f}", f"{kg_time:.1f}",
                                 ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']])
                logger.info("\n" + str(results))

                cur_best_pre_0, cur_stopping_step, should_stop = early_stopping(
                    ret['recall'][0], cur_best_pre_0, cur_stopping_step,
                    expected_order='acc', flag_step=early_stop_step)
                if cur_stopping_step == 0:
                    logger.info("### Found better model at epoch %d", epoch)
                elif should_stop:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

        logger.info("Training complete. Early stopping at epoch %d, recall@20: %.4f", epoch, cur_best_pre_0)

    except Exception as e:
        logger.exception(e)

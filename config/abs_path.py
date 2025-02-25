# -*- coding: utf-8 -*-
# author： Tao Chen
# datetime： 2023/4/4 8:20 
# ide： PyCharm

import platform

if platform.system() == 'Linux':
    if platform.node() == 'lab' or platform.node() == 'zhangd4':
        rafdb = {
            'train': '/home/developers/chentao/datasets/RAF-DB/dataset_train_ct.txt',
            'test': '/home/developers/chentao/datasets/RAF-DB/dataset_test_ct.txt',
        }

        # rafdb = {
        #     'train': '/home/developers/tengjianing/zouwei/dataset/RAF-DB/dataset_train.txt',
        #     'test': '/home/developers/tengjianing/zouwei/dataset/RAF-DB/dataset_test.txt'
        # }

        slgnn = {
            'bitcoinAlpha': {
                'net_train': '/home/developers/chentao/code/bb/slgnn-main/slgnn-main/data/train_test/bitcoinAlpha/bitcoinAlpha_maxC_train{}.edgelist',
                'net_test': '/home/developers/chentao/code/bb/slgnn-main/slgnn-main/data/train_test/bitcoinAlpha/bitcoinAlpha_maxC_test{}.edgelist',
                'features_train': '/home/developers/chentao/code/bb/slgnn-main/slgnn-main/data/features/bitcoinAlpha/bitcoinAlpha_maxC_train{}_features64_tsvd.pkl'},
            'bitcoinOTC': {
                'net_train': '/home/developers/chentao/code/bb/slgnn-main/slgnn-main/data/train_test/bitcoinOTC/bitcoinOTC_maxC_train{}.edgelist',
                'net_test': '/home/developers/chentao/code/bb/slgnn-main/slgnn-main/data/train_test/bitcoinOTC/bitcoinOTC_maxC_test{}.edgelist',
                'features_train': '/home/developers/chentao/code/bb/slgnn-main/slgnn-main/data/features/bitcoinOTC/bitcoinOTC_maxC_train{}_features64_tsvd.pkl'},
            'epinions': {
                'net_train': '/home/developers/chentao/code/bb/slgnn-main/slgnn-main/data/train_test/epinions/epinions_maxC_train{}.edgelist',
                'net_test': '/home/developers/chentao/code/bb/slgnn-main/slgnn-main/data/train_test/epinions/epinions_maxC_test{}.edgelist',
                'features_train': '/home/developers/chentao/code/bb/slgnn-main/slgnn-main/data/features/epinions/epinions_maxC_train{}_features64_tsvd.pkl'},
            'slashdot': {
                'net_train': '/home/developers/chentao/code/bb/slgnn-main/slgnn-main/data/train_test/slashdot/slashdot_maxC_train{}.edgelist',
                'net_test': '/home/developers/chentao/code/bb/slgnn-main/slgnn-main/data/train_test/slashdot/slashdot_maxC_test{}.edgelist',
                'features_train': '/home/developers/chentao/code/bb/slgnn-main/slgnn-main/data/features/slashdot/slashdot_maxC_train{}_features64_tsvd.pkl'},
        }
    elif platform.node() == 'idart_ml':
        slgnn = {
            'bitcoinAlpha': {
                'net_train': '/home/chentao/code/bb/slgnn-main/data/train_test/bitcoinAlpha/bitcoinAlpha_maxC_train{}.edgelist',
                'net_test': '/home/chentao/code/bb/slgnn-main/data/train_test/bitcoinAlpha/bitcoinAlpha_maxC_test{}.edgelist',
                'features_train': '/home/chentao/code/bb/slgnn-main/data/features/bitcoinAlpha/bitcoinAlpha_maxC_train{}_features64_tsvd.pkl'},
            'bitcoinOTC': {
                'net_train': '/home/chentao/code/bb/slgnn-main/data/train_test/bitcoinOTC/bitcoinOTC_maxC_train{}.edgelist',
                'net_test': '/home/chentao/code/bb/slgnn-main/data/train_test/bitcoinOTC/bitcoinOTC_maxC_test{}.edgelist',
                'features_train': '/home/chentao/code/bb/slgnn-main/data/features/bitcoinOTC/bitcoinOTC_maxC_train{}_features64_tsvd.pkl'},
            'epinions': {
                'net_train': '/home/chentao/code/bb/slgnn-main/data/train_test/epinions/epinions_maxC_train{}.edgelist',
                'net_test': '/home/chentao/code/bb/slgnn-main/data/train_test/epinions/epinions_maxC_test{}.edgelist',
                'features_train': '/home/chentao/code/bb/slgnn-main/data/features/epinions/epinions_maxC_train{}_features64_tsvd.pkl'},
            'slashdot': {
                'net_train': '/home/chentao/code/bb/slgnn-main/data/train_test/slashdot/slashdot_maxC_train{}.edgelist',
                'net_test': '/home/chentao/code/bb/slgnn-main/data/train_test/slashdot/slashdot_maxC_test{}.edgelist',
                'features_train': '/home/chentao/code/bb/slgnn-main/data/features/slashdot/slashdot_maxC_train{}_features64_tsvd.pkl'},
        }
    elif platform.node() == 'cn2.sysu' or platform.node() == 'cn5' or platform.node() == 'cn3.sysu':
        slgnn = {
            'bitcoinAlpha': {
                'net_train': '/media/data3/chenhx/slgnn/data/train_test/bitcoinAlpha/bitcoinAlpha_maxC_train{}.edgelist',
                'net_test': '/media/data3/chenhx/slgnn/data/train_test/bitcoinAlpha/bitcoinAlpha_maxC_test{}.edgelist',
                'features_train': '/media/data3/chenhx/slgnn/data/features/bitcoinAlpha/bitcoinAlpha_maxC_train{}_features64_tsvd.pkl'},
            'bitcoinOTC': {
                'net_train': '/media/data3/chenhx/slgnn/data/train_test/bitcoinOTC/bitcoinOTC_maxC_train{}.edgelist',
                'net_test': '/media/data3/chenhx/slgnn/data/train_test/bitcoinOTC/bitcoinOTC_maxC_test{}.edgelist',
                'features_train': '/media/data3/chenhx/slgnn/data/features/bitcoinOTC/bitcoinOTC_maxC_train{}_features64_tsvd.pkl'},
            'epinions': {
                'net_train': '/media/data3/chenhx/slgnn/data/train_test/epinions/epinions_maxC_train{}.edgelist',
                'net_test': '/media/data3/chenhx/slgnn/data/train_test/epinions/epinions_maxC_test{}.edgelist',
                'features_train': '/media/data3/chenhx/slgnn/data/features/epinions/epinions_maxC_train{}_features64_tsvd.pkl'},
            'slashdot': {
                'net_train': '/media/data3/chenhx/slgnn/data/train_test/slashdot/slashdot_maxC_train{}.edgelist',
                'net_test': '/media/data3/chenhx/slgnn/data/train_test/slashdot/slashdot_maxC_test{}.edgelist',
                'features_train': '/media/data3/chenhx/slgnn/data/features/slashdot/slashdot_maxC_train{}_features64_tsvd.pkl'},
        }
elif platform.system() == 'Windows':
    rafdb = {
        'train': 'H:/chent2/datasets/RAF-DB/dataset_train_ct.txt',
        'test': 'H:/chent2/datasets/RAF-DB/dataset_test_ct.txt',
    }

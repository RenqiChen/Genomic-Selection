import os
import numpy as np
import pandas as pd
import random

import torch
from torch.utils.data import Dataset
from kmer_rise3k import kmer_featurization  # import the module kmer_featurization from the kmer.py file

def fetch_dataset(args):

    if args.__contains__("xlsx_path"):
        data_root = args.data_root
        fold_root = args.fold_root
        which_k = args.which_k

        xlsx_path = args.xlsx_path
        task_name = args.task_name
        name2id_path = args.name2id_path
        
        name2id = {}
        with open(name2id_path, "r") as fr:
            for line in fr:
                line = line.strip().split("\t")
                name2id[line[1].replace("_", " ")] = line[0]

        dataframe = pd.read_excel(xlsx_path, sheet_name='Phenotype Data')
        sample_list = dataframe[['NAME', task_name]]

        label_map = {}
        for index, row in sample_list.iterrows():
            if row['NAME'] in name2id.keys() and pd.notna(row[task_name]):
                sample_id = name2id[row['NAME']]
                label = row[task_name]
                label_map[sample_id] = label

        train_d, test_d = apply_KFold(data_root, label_map, fold_root, which_k)

    if args.__contains__("fam_path"):
        data_root = args.data_root
        fam_path = args.fam_path
        which_k = args.which_k

        label_map = {}
        with open(fam_path, "r") as fr:
            contents = fr.readlines()

        for line in contents:
            line = line.strip().split(" ")
            label_map[line[0]] = float(line[5])

        train_d, test_d = split_dataset(data_root, fam_path, which_k)

        if args.normalize:
            mean, std = calculate_mean_and_std(train_d, label_map)
            args.data_mean = mean
            args.data_std = std

    train_D = GENE(train_d, label_map, args)
    test_D = GENE(test_d, label_map, args)
    return train_D, test_D


def hierarchical_sampling(data, which_k, sample_ratio=0.2):
    val_list = set(data.values())
    val2list = {v:[] for v in val_list}


    for key, val in data.items():
        val2list[val].append(key)

    train_list = []
    test_list = []

    for v in val_list:
        size = len(val2list[v])
        random.shuffle(val2list[v])

        st_index = int(size * sample_ratio * (which_k - 1))
        ed_index = int(size * sample_ratio * which_k)
        test_files = val2list[v][st_index:ed_index]
        train_files = list(set(val2list[v]) - set(test_files))
        
        train_list += train_files
        test_list += test_files

    return train_list, test_list



def apply_KFold(data_root, label_map, fold_root, which_k):
    split_file_path = os.path.join(fold_root, str(which_k))
    
    if os.path.exists(split_file_path):
        train_d, test_d = [], []
        with open(split_file_path, "r") as fr:
            for line in fr:
                line = line.strip().split(" ")
                if line[-1] == '1':
                    train_d.append(os.path.join(data_root, line[0]+".txt"))
                else:
                    test_d.append(os.path.join(data_root, line[0]+".txt"))

    else:
        sampled_train, sampled_test = hierarchical_sampling(label_map, which_k)
        train_d, test_d = [], []

        with open(split_file_path, "w") as fw:
        
            for file_name in sampled_train:
                train_d.append(os.path.join(data_root, file_name+".txt"))
                fw.write(file_name + " " + "1\n")
        
            for file_name in sampled_test:
                test_d.append(os.path.join(data_root, file_name+".txt"))
                fw.write(file_name + " " + "0\n")

    return train_d, test_d


def calculate_mean_and_std(train_file, label_map):
    data = [] 
    for file_path in train_file:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        label = label_map[file_name]
        data.append(label)
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    return mean, std


def split_dataset(data_root, fam_path, which_k):
    with open(fam_path, "r") as fr:
        contents = fr.readlines()

    is_train = {}
    for line in contents:
        line = line.strip().split(" ")
        if line[5 + which_k] != "NA":
            is_train[line[0]] = True
        else:
            is_train[line[0]] = False
    
    file_lists = os.listdir(data_root)

    train_d, test_d = [], []
    for file_name in file_lists:
        if os.path.splitext(file_name)[0] not in is_train:
            continue
        if is_train[os.path.splitext(file_name)[0]]:
            train_d.append(os.path.join(data_root, file_name))
        else:
            test_d.append(os.path.join(data_root, file_name))

    return train_d, test_d

class GENE(Dataset):
    
    def __init__(self, file_paths, label_map, args):
        self.file_paths = file_paths
        self.label_map = label_map
        self.args = args
        self.k = self.args.kmer
        self.obj = kmer_featurization(self.k)

        self.tokenizer = CharacterTokenizer(args.vocabs)

        self.uni_map = np.unique(list(val for val in self.label_map.values()))
        self.uni_label_map = {}
        for i, v in enumerate(self.uni_map):
            self.uni_label_map[v] = i

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        file_path = self.file_paths[index]
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        with open(file_path, "r") as fr:
            contents = fr.read()

        mapping = {"N": "X", "Y": "X", "K": "X", "W": "X", "R": "X", "S": "X", "M": "X"}

        translation_table = str.maketrans(mapping)
        contents = contents.translate(translation_table)
        x = self.tokenizer.tokenize(contents)
        sequence_str = ''.join(map(str, x))
        # True: overlap
        kmer_feature = self.obj.obtain_kmer_feature_for_one_sequence(sequence_str, overlapping=False)
        x=np.array(kmer_feature)
        # Random Mask
        # 设置掩码比例（此处为50%）
        mask_ratio = 0.45

        # 生成一个与数组相同形状的随机布尔掩码
        mask = np.random.rand(len(x)) < mask_ratio

        # 将掩码为True的位置的元素置于末尾
        x[mask] = 5 ** self.k
        if file_name in self.label_map:
            label = self.label_map[file_name]
        if self.args.normalize:
            label = (label - self.args.data_mean) / self.args.data_std
            label = np.array(label, dtype=np.float32)

        if self.args.label_unique:
            label = self.uni_label_map[label]
            label = np.array(label, dtype=np.float32)
        return {"input":x, "label":label}
    

class CharacterTokenizer:

    def __init__(self, characters):
        self._vocab_str_to_int = {
            "PAD" : 0,
            **{ch: i+1 for i, ch in enumerate(characters)},
            "MASK": len(characters)+1,
            "UNK": len(characters)+2
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        
    def tokenize(self, text):
        tokens = []
        for char in text:
            if char not in self._vocab_str_to_int:
                tokens.append(self._vocab_str_to_int["UNK"])
            else:
                tokens.append(self._vocab_str_to_int[char])
        
        tokens = np.array(tokens, dtype=np.int32)
        return tokens

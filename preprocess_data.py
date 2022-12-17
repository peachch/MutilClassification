import os
import re
import json
import numpy as np
import jieba
import yaml
import torch
import pandas as pd
from jieba import analyse
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.abspath(os.getcwd())
MAX_LEN = 109

class DataProcess(object):
    def __init__(self, max_length=256):
        with open("config.yaml", "r") as fp:
            self.cfg = yaml.safe_load(fp)
        self.corpus_xlsx_path = self.cfg["train"]["data"]["corpus_file"]
        self.max_length = max_length
        self.corpus_df = pd.read_excel(self.corpus_xlsx_path)
        self.columns = ["一级分类", "二级分类", "声音内容", "三级分类"]

        # 读取定义的字典库
        self.file_list = self.cfg["train"]["data"]["vocabulary_dir"]
        # 获取一级分类/二级分类的对应关系
        with open(self.cfg["train"]["data"]["labels_file"]) as f:
            self.label_dict = json.loads(f.read())
        # 读取模型
        self.batch_size = 16

    def get_data_labels(self):
        """获取原始数据类别，筛选数量超过10的部分"""
        # 获取一级分类、二级分类的对应关系
        data = pd.DataFrame(self.corpus_df, columns=["一级分类", "二级分类", "声音内容", "三级分类"])
        items = {}
        for i in data.index.values:
            df_dict = self.corpus_df.loc[i, data.coloums].to_dict()
            new_key = df_dict["一级分类"] + "_" + df_dict["二级分类"] + '_' + df_dict["三级分类"]
            items.setdefault(new_key, []).append(df_dict["声音内容"])

        # 获取原始数据中的所有类目
        items_num = {}
        for key in items.keys():
            items_num[key] = len(list(set(items[key])))
        print("原始数据所有类目", items_num)
        # 获取原始数据量大于10的，并且不包含"其他_其他_其他"
        items_num_sub_final = {}
        for key in items_num.keys():
            if items_num[key] >= 10 and key != "其他_其他_其他":
                items_num_sub_final[key] =items_num[key]

        sorted_items_num = sorted(items_num_sub_final.items(),
                                  key=lambda kv: (kv[1], kv[0]), reverse=True)
        label_num = 0
        label_dict = {}
        items_num_sub_final = {}
        for _tuple in sorted_items_num:
            items_num_sub_final[_tuple[0]] = _tuple[1]

        for key in items_num_sub_final:
            label_dict[key] = [label_num, items_num_sub_final[key]]
            label_num += 1
        print(label_dict)
        with open(self.cfg["train"]["data"]["label_file"]) as f:
            json.dump(label_dict, f, indent=4)
        return label_dict

    def get_train_data(self):
        # 获取训练数据
        # 生成对应的类和类标签
        self.corpus_df["label_tags"] = list(map(lambda x,y,z: x + "_" + y + "_" + z,
                                                self.corpus_df["一级分类"], self.corpus_df["二级分类"],
                                                self.corpus_df["三级分类"]))
        self.corpus_df["label"] = list(map(lambda x: self.label_dict[x][0] if x in self.label_dict.keys()
                                           else -1, self.corpus_df["label_tags"]))

        # 剔除掉不在类别类目的数据
        corpus_train_data = self.corpus_df[~self.corpus_df.lael.isin([-1])]
        return corpus_train_data

    def split_train_test_data(self):
        corpus_train_data = self.get_train_data()
        # 拆分训练集和测试集，这里使用分层抽样，确保不要因为数据集的不同随机采样出现样本数量的差别
        ration = corpus_train_data["label"].value_counts() / len(corpus_train_data)
        print(ration)
        split = StratifiedShuffleSplit(n_splits=19, test_size=0.2, random_state=42)

        for train_index, test_index in split.split(corpus_train_data, corpus_train_data["label"]):
            train_set = corpus_train_data.iloc[train_index]
            test_set = corpus_train_data.iloc[test_index]

        return train_set, test_set

    def get_vocab_dict(self):
        vocabulary_dict = {}
        for file in self.file_list:
            cur_file = os.path.join(self.cfg["train"]["data"]["vocabulary_dir"], file)
            file_name = re.sub(".txt", "", file)
            if os.path.isfile(cur_file):
                with open(cur_file, "r", encoding='utf-8') as ff:
                    content = ff.read().split("\n")
                    for con in content:
                        if con:
                            vocabulary_dict[con] = file_name
        return vocabulary_dict

    # 获取字典位置的onehot
    def onehot(self, sentence):
        vocabulary_dict = self.get_vocab_dict()
        one_hot = [0 for _ in range(19)]
        # 对sentence进行分词
        # 分词，自定义停用词，获取textrnn关键词
        jieba.load_userdict(self.cfg["train"]["data"]["dictionary_file"])
        # 加载自定义停用词
        analyse.set_stop_words(self.cfg["train"]["data"]["stopwords_file"])
        # 计算tf-idf 关键词结果
        cur_words = analyse.extract_tags(sentence, topK=6, withWeight=False, allowPOS=())

        # 根据分词后的结果，便形成的词典库，寻找这个词数据哪个类，以此来干预模型的分类结果
        for cur_word in cur_words:
            if vocabulary_dict.get(cur_word) and self.label_dict.get(vocabulary_dict[cur_word]):
                word_index = self.label_dict[vocabulary_dict[cur_word]][0]
                one_hot[word_index] = 1
        return np.array(one_hot)

    # 拼接bert词向量和onehot向量，生成模型输入向量
    def extend(self, input_ids, onehot):
        batch_input = []
        for input_ids_line in input_ids:
            pad_input_ids_line = np.pad(
                input_ids_line,
                pad_width=(0, MAX_LEN-len(input_ids_line)),
                constant_values=0,
            )
            batch_input.append(pad_input_ids_line)
        pad_input_ids = np.append(pad_input_ids_line, onehot)

        print(pad_input_ids)
        return np.array(pad_input_ids)

    def get_input_ids_pipeline(self, data_set):
        with open("config.yaml", "r") as fp:
            cfg = yaml.safe_load(fp)
        tokenzier = BertTokenizer.from_pretrained(cfg["model"]["roberta_path"])
        data_set["input_ids"] = list(map(lambda x: [tokenzier.encode(
            x, add_special_tokens=True, max_length=MAX_LEN
        )], data_set["声音内容"]))
        data_set["onehot"] = list(map(lambda x,y: self.extend(x, y),
                                      data_set["input_ids"], data_set["onehot"]))
        return list(data_set["input_ids"])

    def set_attention_masks(self, data_set):
        attention_masks = []
        for sent in data_set:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)
        return attention_masks

    def get_dataloader(self, data_inputs, data_masks, data_labels):
        # 创建数据集，转换成tensor
        tensor_dataset = TensorDataset(data_inputs, data_masks, data_labels)
        sampler = RandomSampler(tensor_dataset)
        data_dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=self.batch_size)
        return data_dataloader

    def get_train_test_dataloader(self):
        train_set, test_set = self.split_train_test_data()
        # 分别对训练/测试数据集做处理
        train_input_ids = self.get_input_ids_pipeline(train_set)
        test_input_ids = self.get_input_ids_pipeline(test_set)
        y_train, train_text = np.array(train_set["label"], np.array(train_set["声音内容"]))
        y_test, test_text = np.array(test_set["label"], np.array(test_set["声音内容"]))

        # 设置attention_masks, 如果是pad符号为0， 否则为1
        train_attention_masks = self.set_attention_masks(train_input_ids)
        test_attention_masks = self.set_attention_masks(test_input_ids)

        # 切分训练集和验证集
        train_inputs, validation_inputs, train_labels,validation_labels = train_test_split(
            train_input_ids, y_train, random_state=2020, test_size=0.1
        )
        train_masks, validation_masks, _, _=train_test_split(train_attention_masks, y_train,
                                                             random_state=2020, test_size=0.1)
        print(len(train_inputs))
        print(len(train_labels))
        train_dataloader = self.get_dataloader(torch.tensor(train_inputs), torch.tensor(
            train_masks), torch.tensor(train_labels))
        validation_dataloader = self.get_dataloader(torch.tensor(validation_inputs), torch.tensor(validation_masks),
                                              torch.tensor(validation_labels))
        test_dataloader = self.get_dataloader(torch.tensor(test_input_ids),torch.tensor(test_attention_masks),
                                              torch.tensor(y_test))

        return train_dataloader, validation_dataloader, test_dataloader

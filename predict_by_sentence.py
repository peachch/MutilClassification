import json
import os
import re
from collections import Counter

import jieba
import numpy as np
import torch
import yaml
from jieba import analyse
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from utils.logger import get_logger
from preprocess_data import DataProcess
from transformers import BertForSequenceClassification, BertTokenizer

class Predict(object):
    def __init__(self, cfg):
        self.model_path = cfg["inference"]["ckpt_path"]
        self.MAX_LEN = 128
        self.batch_size = 16
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)

        with open(cfg["train"]["data"]["label_file"]) as labels:
            self.label_dict = json.loads(labels.read())
        self.index_label = []
        for key in self.label_dict:
            self.index_label.append(key)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_path,
            num_labeld = 19,
            output_attentions = False,
            output_hidden_states = False
        )
        self.model.to(device)
        self.model.eval()
        logger.info("OK")

    def get_dataloader(self, sentence):
        test_input_ids = [
            self.tokenizer.encode(
                sentence, add_special_tokens = True, max_length = self.MAX_LEN
            )
        ]
        # 设置attention_masks, 如果是pad符号则为0， 否则为1
        test_attention_masks = []
        for sent in test_input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            test_attention_masks.append(att_mask)
        test_input = torch.tensor(test_input_ids)
        test_masks = torch.tensor(test_input_ids)

        # Create the Dataloader dor our test set
        test_data = TensorDataset(test_input, test_masks)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=self.batch_size
        )
        return test_dataloader

    def model_test(self, sentence_dataloader):
        for batch in sentence_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch

            with torch.no_grad():
                outputs = self.model(
                b_input_ids, token_type_ids = None,
                attention_mask = b_input_mask
                )

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predict_label = np.argmax(logits, axis=1).flatten()
        return predict_label

    def get_keywords(self, sentence):
        """得到句子的分词结果

        :param sentence: string
        :return: list
        """
        dictionary_path = cfg["train"]["data"]["dictionary_file"]
        stopwords_path = cfg["train"]["data"]["stop_words_file"]
        jieba.load_userdict(dictionary_path)
        analyse.set_stop_words(stopwords_path)
        # 计算tf-idf 关键词结果
        cur_words = analyse.extract_tags(
            sentence, topK=10, withWeight=False, allowPOS=()
        )
        return cur_words

    def get_emb(self):
        """读取字典数据

        :return: dict
        """
        vocabulary_path = cfg["train"]["data"]["vocabulary_dir"]
        file_list = os.listdir(vocabulary_path)
        vocabulary_dict = {}
        for file in file_list:
            cur_file = os.path.join(vocabulary_path, file)
            file_name = re.sub(".txt", "", file)
            if os.path.isfile(cur_file):
                with open(cur_file, "r", encoding="utf-8") as ff:
                    content = ff.read().split("\n")
                    for con in content:
                        if con:
                            vocabulary_dict[con] = file_name
        return vocabulary_dict

    def keywords_get_label(self, keywords):
        """根据分词后的结果，遍历形成词典库，寻找这个词数据哪个类，以此来干预模型
        的分类结果

        :param keywords: list
        :return:
        """
        predict_label = []
        vocabulary_dict = self.get_emb()
        for cur_word in keywords:
            if vocabulary_dict.get(cur_word) and self.label_dict.get(
                vocabulary_dict[cur_word]
            ):
                word_index = self.label_dict[vocabulary_dict[cur_word]][0]
                predict_label.append(word_index)
                logger.info("循环里的预测", predict_label, cur_word)
        if len(predict_label) > 1:
            logger.info("所有的预测结果", predict_label, keywords)

        "如果去重后只包含一个类，则返回此类"
        if len(set(predict_label)) == 1:
            return predict_label[0]
        labels_count = Counter(predict_label)
        sorted_labels_count = sorted(labels_count.items(), key=lambda kv:kv[1], reverse=True)
        """如果排序后的多个类不相同，则返回最大值"""
        if (
            sorted_labels_count
            and sorted_labels_count[0][1] != sorted_labels_count[1][1]
        ):
            return sorted_labels_count[0][0]
        else:
            return -1

    def predict(self, text):
        """预测

        """
        keywords = self.get_keywords(text)
        predict_label = self.keywords_get_label(keywords)
        if predict_label == -1:
            test_dataloder = self.get_dataloader(text)
            predict_label = self.model_test(test_dataloder)[0]
        logger.info(
            {
            "用户声音": text,
            "预测label": predict_label,
            "预测结果": self.index_label[predict_label]
            }
        )
        return predict_label, self.index_label[predict_label]


if __name__ == "__main__":
    with open("config.yaml", "r") as fp:
        cfg = yaml.safe_load(fp)
    log_dir = cfg["trian"]["log"]["log_dir"]
    logger = get_logger(
        "train",
        log_dir = log_dir,
        log_filename = cfg["train"]["log"]["log_filename"],
    )
    device = cfg["train"]["device"]
    dataset_pre = DataProcess()
    predict = Predict(cfg)
    predict.predict("this is a classication text")
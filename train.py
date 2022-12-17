import os
import time
import datetime
import numpy as np
import yaml
import torch
from model import Model
from transformers import AdamW, get_linear_schedule_with_warmup
from preprocess_data import DataProcess
from utils.logger import get_logger

BASE_DIR = os.path.abspath(os.getcwd())
MAX_LEN = 109
epochs = 4

def format_time(elapsed):
    elapsed_rouded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rouded))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat==labels_flat)/len(labels_flat)

def train_piplines(train_dataloader, validation_dataloader):
    device = cfg["train"]["device"]
    model = Model(**cfg["model"])
    model.to_device()

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_value = []
    for epoch_i in range(0, epochs):
        logger.info("")
        logger.info('======== Epoch {:}/{:} ========='.format(epoch_i+1, epochs))
        logger.info('Training...')

        # 记录每个epoch时间
        t0 = time.time()

        # 重置total loss
        total_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                logger.info('Batch {:>5} of {:>5}, Elapsed:{:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            inputs = {"input_ids": b_input_ids, "attention_mask": b_input_mask, "labels": b_labels}

            model.zero_grad()
            outputs = model(inputs)
            loss = outputs[0]
            print(loss)
            total_loss += loss.time()
            # 计算gradients
            loss.backward()

            # 切分norm of the gradients to 1.0 防止梯度爆炸
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # 计算训练数据的平均损失
        avg_train_loss = total_loss / len(train_dataloader)
        loss_value.append(avg_train_loss)

        logger.info("")
        logger.info("Average training loss: {0:.2f}".format(avg_train_loss))
        logger.infor("Training epcoh took:{:}".format(format_time(time.time() - t0)))

        logger.info("")
        logger.info("Running Validation...")

        t0 = time.time()

        # during evaluation
        model.eval()
        eval_loss, eval_accuray = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            inputs = {"input_ids": b_input_ids, "attention_mask": b_input_mask}
            with torch.no_grad():
                outputs = model(inputs)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            # 计算测试集准确率
            tmp_eval_accurancy= flat_accuracy(logits, label_ids)

            # total accuracy
            eval_accuray += tmp_eval_accurancy
            nb_eval_steps += 1

        model.train()

        # Report the final accuray for this validation run
        logger.info("Accuracy:{0:.2f}".format(eval_accuray / nb_eval_steps))
        logger.info("Validation took:{:}".format(format_time(time.time()- t0)))

    logger.info("")
    logger.info("Training complete")

if __name__=="__main__":
    with open("config.yaml", "r") as fp:
        cfg = yaml.safe_load(fp)
    log_dir = cfg["train"]["log"]["log_dir"]
    logger = get_logger(
        "train",
        log_dir=log_dir,
        log_filename=cfg["train"]["log"]["log_filename"],
    )
    corpus_path = cfg["train"]["data"]["corpus_file"]
    train_dataloader, validation_dataloader, _,_ = DataProcess().get_train_test_dataloader()
    train_piplines(train_dataloader, validation_dataloader)
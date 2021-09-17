# coding=utf-8
import random
import os

import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import time

import yaml
import torch
import torch.nn as nn

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam

from utils import get_device,load_data,classifiction_metric,evaluate
from model import BertATT

def train(epoch_num, n_gpu, model, train_dataloader, dev_dataloader,
          optimizer, criterion, gradient_accumulation_steps, device, label_list,
          output_model_file, output_config_file, log_dir, print_step, early_stop):
    print("start to train the model")
    early_stop_times = 0

    writer = SummaryWriter(
        log_dir=log_dir + '/' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time())))

    best_dev_loss = float('inf')
    best_auc = 0
    best_acc = 0

    global_step = 0
    for epoch in range(int(epoch_num)):
        if early_stop_times >= early_stop:
            break
        print(f'---------------- Epoch: {epoch+1:02} ----------')
        epoch_loss = 0
        train_steps = 0

        all_preds = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batchData = tuple(t.to(device) for t in batch)
            _, input_ids, input_mask, segment_ids, label_ids = batchData

            logits = model(input_ids, segment_ids, input_mask, labels=None)
            loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

            """ 修正 loss """
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            train_steps += 1
            loss.backward()

            # 用于画图和分析的数据
            epoch_loss += loss.item()
            preds = logits.detach().cpu().numpy()
            outputs = np.argmax(preds, axis=1)
            all_preds = np.append(all_preds, outputs)
            label_ids = label_ids.to('cpu').numpy()
            all_labels = np.append(all_labels, label_ids)

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % print_step == 0 and global_step != 0:

                    """ 打印Train此时的信息 """
                    train_loss = epoch_loss / train_steps
                    train_acc, train_report, train_auc = classifiction_metric(all_preds, all_labels, label_list)

                    dev_loss, dev_acc, dev_report, dev_auc = evaluate(model, dev_dataloader, criterion, device,
                                                                      label_list)

                    c = global_step // print_step
                    writer.add_scalar("loss/train", train_loss, c)
                    writer.add_scalar("loss/dev", dev_loss, c)

                    writer.add_scalar("acc/train", train_acc, c)
                    writer.add_scalar("acc/dev", dev_acc, c)

                    writer.add_scalar("auc/train", train_auc, c)
                    writer.add_scalar("auc/dev", dev_auc, c)

                    for label in label_list:
                        writer.add_scalar(label + ":" + "f1/train", train_report[label]['f1-score'], c)
                        writer.add_scalar(label + ":" + "f1/dev",
                                          dev_report[label]['f1-score'], c)

                    print_list = ['macro avg', 'weighted avg']
                    for label in print_list:
                        writer.add_scalar(label + ":" + "f1/train",
                                          train_report[label]['f1-score'], c)
                        writer.add_scalar(label + ":" + "f1/dev",
                                          dev_report[label]['f1-score'], c)

                    # 以 acc 取优
                    if dev_acc > best_acc:
                        best_acc = dev_acc

                        # 以 auc 取优
                        # if dev_auc > best_auc:
                        #     best_auc = dev_auc

                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        torch.save(model_to_save.state_dict(), output_model_file)
                        with open(output_config_file, 'w') as f:
                            f.write(model_to_save.config.to_json_string())

                        early_stop_times = 0
                    else:
                        early_stop_times += 1
    writer.close()


if __name__ == "__main__":
    print("This is the start of the main program")

    configFile = "./config.yaml"
    label_list = ['0', '1']

    with open(configFile) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("config ", config["output_dir"])

    if not os.path.exists(config["output_dir"] + config["model_name"]):
        os.makedirs(config["output_dir"] + config["model_name"])

    if not os.path.exists(config["cache_dir"] + config["model_name"]):
        os.makedirs(config["cache_dir"] + config["model_name"])

    # Bert 模型输出文件
    output_model_file = os.path.join(config["output_dir"], config["model_name"], WEIGHTS_NAME)
    output_config_file = os.path.join(config["output_dir"], config["model_name"],CONFIG_NAME)

    # 设备准备
    gpu_ids = [int(device_id) for device_id in config["gpu_ids"].split()]

    device, n_gpu = get_device(gpu_ids[0])
    if n_gpu > 1:
        n_gpu = len(gpu_ids)

    config["train_batch_size"] = config["train_batch_size"] // config["gradient_accumulation_steps"]

    # 设定随机种子
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config["seed"])

    # 数据准备
    tokenizer = BertTokenizer.from_pretrained(
        config["bert_vocab_file"], do_lower_case=config["do_lower_case"])  # 分词器选择

    num_labels = len(label_list)

    #Load data
    train_dataloader, train_examples_len = load_data(
        config["data_dir"], tokenizer, config["max_seq_length"], config["train_batch_size"], "train", label_list)

    num_train_optimization_steps = int(
        train_examples_len / config["train_batch_size"] / config["gradient_accumulation_steps"]) * config["num_train_epochs"]

    model = BertATT.from_pretrained(
        config["bert_model_dir"], cache_dir=config["cache_dir"], num_labels=num_labels)

    model.to(device)

    """ 优化器准备 """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=float(config["learning_rate"]),
                         warmup=float(config["warmup_proportion"]),
                         t_total=int(num_train_optimization_steps))

    """ 损失函数准备 """
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    train(config["num_train_epochs"], n_gpu, model, train_dataloader, train_dataloader, optimizer,
          criterion, config["gradient_accumulation_steps"], device, label_list, output_model_file, output_config_file,
          config["log_dir"], config["print_step"], config["early_stop"])





























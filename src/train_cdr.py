import argparse
import os

import numpy as np
import torch
# from apex import amp
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import get_constant_schedule_with_warmup             
from model import DocREModel
from utils import set_seed, collate_fn, assign_distance_bucket
#from prepro import read_cdr, read_gda, read_docred, read_docred_con
from convert_pro import read_docred_con
from adj_utils import convert_3dsparse_to_4dsparse      
# import wandb
from time import time                           



def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = 875
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)       

        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            t1 = time()                 
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                adjacency = convert_3dsparse_to_4dsparse(batch[5]).to(args.device)
                # sub_adjacency = convert_3dsparse_to_4dsparse(batch[10]).to(args.device)
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          'adjacency': adjacency,        
                          'link_pos': batch[6],     
                          'nodes_info': batch[7],
                          # 'sub_nodes': batch[9],
                          # 'sub_adjacency': sub_adjacency,
                          }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    # test_score, test_output = evaluate(args, model, test_features, tag="test")
                    t2 = time()                 
                    print(f'epoch:{epoch}, time:{humanized_time(t2-t1)}, loss:{loss}\n')
                    print(dev_output)
                    # print(test_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)
                            with open('./saved_model/CDR_new/log.txt', 'a') as f:
                                f.writelines(f'epoch:{epoch}\n')
                                f.writelines(f'{dev_output}\n')
                                # f.writelines(f'{test_output}\n')
                                f.writelines('\n')

        return num_steps

    new_layer = ["extractor", "bilinear", "linear", "rgcn", "reason", "cc_module"]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},           
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds, golds, dists, ent_dis= [], [], [], []
    for batch in dataloader:
        model.eval()
        adjacency = convert_3dsparse_to_4dsparse(batch[5]).to(args.device)
        # sub_adjacency = convert_3dsparse_to_4dsparse(batch[10]).to(args.device)
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'adjacency': adjacency,       
                  'link_pos': batch[6],     
                  'nodes_info': batch[7],
                  # 'sub_nodes': batch[9],
                  # 'sub_adjacency': sub_adjacency,
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            golds.append(np.concatenate([np.array(label, np.float32) for label in batch[2]], axis=0))
            dists.append(np.concatenate([np.array(dist, np.float32) for dist in batch[8]], axis=0))
            ent_dis.append(np.concatenate([np.array(dist, np.float32) for dist in batch[9]], axis=0))

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    golds = np.concatenate(golds, axis=0).astype(np.float32)
    dists = np.concatenate(dists, axis=0).astype(np.float32)
    ent_dis = np.concatenate(ent_dis, axis=0).astype(np.float32)

    tp = ((preds[:, 1] == 1) & (golds[:, 1] == 1)).astype(np.float32).sum() #预测正确的
    fn = ((golds[:, 1] == 1) & (preds[:, 1] != 1)).astype(np.float32).sum() #正确但是预测错误的
    fp = ((preds[:, 1] == 1) & (golds[:, 1] != 1)).astype(np.float32).sum() #错误但是预测正确的
    tn = ((preds[:, 1] == 0) & (golds[:, 1] == 0)).astype(np.float32).sum()
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    fer = tn / (tn + fp + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)

    tp_intra = ((preds[:, 1] == 1) & (golds[:, 1] == 1) & (dists == 0)).astype(np.float32).sum()
    fn_intra = ((golds[:, 1] == 1) & (preds[:, 1] != 1) & (dists == 0)).astype(np.float32).sum()
    fp_intra = ((preds[:, 1] == 1) & (golds[:, 1] != 1) & (dists == 0)).astype(np.float32).sum()
    precision_intra = tp_intra / (tp_intra + fp_intra + 1e-5)
    recall_intra = tp_intra / (tp_intra + fn_intra + 1e-5)
    f1_intra = 2 * precision_intra * recall_intra / (precision_intra + recall_intra + 1e-5)

    tp_inter = ((preds[:, 1] == 1) & (golds[:, 1] == 1) & (dists == 1)).astype(np.float32).sum()
    fn_inter = ((golds[:, 1] == 1) & (preds[:, 1] != 1) & (dists == 1)).astype(np.float32).sum()
    fp_inter = ((preds[:, 1] == 1) & (golds[:, 1] != 1) & (dists == 1)).astype(np.float32).sum()
    precision_inter = tp_inter / (tp_inter + fp_inter + 1e-5)
    recall_inter = tp_inter / (tp_inter + fn_inter + 1e-5)
    f1_inter = 2 * precision_inter * recall_inter / (precision_inter + recall_inter + 1e-5)

    # 根据实体距离，检查模型性能，根据每个实体的第一个提及出现位置之间距离来将数据集分成不同的子集
    # 示例边界值，根据您的需求定义不同的距离区间
    distance_buckets = [8, 32, 64, 128]
    # 为每个样本的距离分配区间标签
    bucket_labels = np.array([assign_distance_bucket(d, distance_buckets) for d in ent_dis])

    # 然后基于这些标签来统计每个区间的TP, FP, FN
    buckets = len(distance_buckets) + 1  # 区间数+1，最后一个区间是>=最大边界值
    dis_tp = np.zeros(buckets, dtype=np.float32)
    dis_fp = np.zeros(buckets, dtype=np.float32)
    dis_fn = np.zeros(buckets, dtype=np.float32)

    for pred, gold, bucket in zip(preds[:, 1], golds[:, 1], bucket_labels):
        if pred == gold == 1:
            dis_tp[bucket] += 1
        elif pred == 1 and gold != 1:
            dis_fp[bucket] += 1
        elif pred != 1 and gold == 1:
            dis_fn[bucket] += 1
    dis_precision = dis_tp / (dis_tp + dis_fp + 1e-5)  # 防止分母为0
    dis_recall = dis_tp / (dis_tp + dis_fn + 1e-5)  # 防止分母为0
    dis_f1_scores = 2 * dis_precision * dis_recall / (dis_precision + dis_recall + 1e-5)  # 防止分母为0


    output = {
        "{}_p".format(tag): precision * 100,
        "{}_r".format(tag): recall * 100,
        "{}_fer".format(tag): fer * 100,
        "{}_f1".format(tag): f1 * 100,
        "{}_f1_intra".format(tag): f1_intra * 100,
        "{}_f1_inter".format(tag): f1_inter * 100,
        "{}_f1_1".format(tag): dis_f1_scores[0] * 100,
        "{}_f1_2".format(tag): dis_f1_scores[1] * 100,
        "{}_f1_3".format(tag): dis_f1_scores[2] * 100,
        "{}_f1_4".format(tag): dis_f1_scores[3] * 100,
        "{}_f1_5".format(tag): dis_f1_scores[4] * 100,
    }
    return f1, output

def humanized_time(second):
    """
    :param second: time in seconds
    :return: human readable time (hours, minutes, seconds)
    """
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/cdr/convert_CDR", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="./biobert_base", type=str)
    parser.add_argument("--train_file", default="convert_train.json", type=str)
    parser.add_argument("--dev_file", default="convert_dev.json", type=str)
    parser.add_argument("--test_file", default="convert_test.json", type=str)
    parser.add_argument("--save_path", default="./saved_model/CDR_noHTG/best.model", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_entity_number", default=35, type=int,
                        help="the max entity number in dataset.")                           

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=1, type=int,
                        help="Max number of labels in the prediction.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization.")
    parser.add_argument("--num_class", type=int, default=2,
                        help="Number of relation types in dataset.")

    args = parser.parse_args()
    # wandb.init(project="CDR")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    #read = read_cdr if "cdr" in args.data_dir else read_gda
    read = read_docred_con

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = read(train_file, './dataset/cdr/processed/train_cache', tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read(dev_file, './dataset/cdr/processed/dev_cache', tokenizer, max_seq_length=args.max_seq_length)
    test_features = read(test_file, './dataset/cdr/processed/test_cache', tokenizer, max_seq_length=args.max_seq_length)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    # model = DocREModel(config, model, num_labels=args.num_labels)
    model = DocREModel(config, model, num_labels=args.num_labels, max_entity=args.max_entity_number)            
    model.to(args.device)

    if args.load_path == "":
        train(args, model, train_features, dev_features, test_features)
    else:
        # model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print(dev_output)
        print(test_output)


if __name__ == "__main__":
    main()

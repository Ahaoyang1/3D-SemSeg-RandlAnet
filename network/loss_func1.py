import torch
from torch import random
import random
import numpy as np
from utils.config import ConfigDs as cfg

# def compute_loss(end_points, dataset, criterion, selected_num= 100, sample_weight=None):
#
#     logits = end_points['logits']
#     labels = end_points['labels']
#     embeddings = end_points['embeddings']
#
#
#     logits = logits.transpose(1, 2).reshape(-1, dataset.num_classes)
#     # print('logits',logits.shape)
#     labels = labels.reshape(-1)
#     embeddings=embeddings.squeeze()
#     # print(embeddings.shape, logits.shape)
#     embeddings = embeddings.transpose(1, 2).reshape(-1, cfg.embedding_size)
#
#     # Boolean mask of points that should be ignored
#     ignored_bool = (labels == 0)
#
#     for ign_label in dataset.ignored_labels:
#         ignored_bool = ignored_bool | (labels == ign_label)
#     # print('ignored_bool',ignored_bool.shape)
#     # Collect logits and labels that are not ignored
#     valid_idx = ignored_bool == 0
#     # print('valid_idx',valid_idx.shape)
#     valid_logits = logits[valid_idx, :]
#     valid_labels_init = labels[valid_idx]
#     embeddings=embeddings[valid_idx,:]
#
#     # Reduce label values in the range of logit shape
#     reducing_list = torch.arange(0, dataset.num_classes).long().to(logits.device)
#     inserted_value = torch.zeros((1,)).long().to(logits.device)
#     for ign_label in dataset.ignored_labels:
#         reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
#     valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
#     # print(valid_labels.shape,torch.min(valid_labels),torch.max(valid_labels))
#
#     # 计算交叉熵损失
#     cls_loss = criterion['nll'](valid_logits, valid_labels).mean()
#     end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels
#     end_points['cls_loss'] = cls_loss
#
#     # 打印分类损失
#     print(f"Classification Loss:{cls_loss.item()}")
#
#     # 计算嵌入损失
#     if selected_num != 0:
#         score = valid_logits.max(dim=1)[1]
#         mask = score == valid_labels
#         embeddings = embeddings[mask, :]
#         valid_labels = valid_labels[mask]
#         N, _ = embeddings.shape
#
#         index = torch.LongTensor(random.sample(range(N), selected_num))
#         index = index.to(logits.device)
#         embeddings = torch.index_select(embeddings, 0, index)
#         valid_labels = torch.index_select(valid_labels, 0, index)
#
#         # 计算嵌入损失
#         embedding_loss = criterion['discriminative'](embeddings, valid_labels,32)
#         end_points['embedding_loss'] = embedding_loss.mean()
#
#         # 打印嵌入损失
#         print(f"Embedding Loss: {end_points['embedding_loss'].item()}")
#
#         # 总损失
#         total_loss = cls_loss + end_points['embedding_loss']
#     else:
#         total_loss = cls_loss
#
#     # 打印总损失
#     print(f"Total Loss: {total_loss.item()}")
#     end_points['total_loss'] = total_loss
#     return end_points


def compute_loss(end_points, dataset, criterion, selected_num= 0, sample_weight=None):

    logits = end_points['logits']
    labels = end_points['labels']
    embeddings = end_points['embeddings']


    logits = logits.transpose(1, 2).reshape(-1, dataset.num_classes)
    # print('logits',logits.shape)
    labels = labels.reshape(-1)
    embeddings=embeddings.squeeze()
    # print(embeddings.shape, logits.shape)
    embeddings=embeddings.transpose(1,2).reshape(-1, 32)

    # Boolean mask of points that should be ignored
    ignored_bool = (labels == 100)
    for ign_label in dataset.ignored_labels:
        ignored_bool = ignored_bool | (labels == ign_label)
    # print('ignored_bool',ignored_bool.shape)
    # Collect logits and labels that are not ignored
    valid_idx = ignored_bool == 0
    # print('valid_idx',valid_idx.shape)
    valid_logits = logits[valid_idx, :]
    valid_labels_init = labels[valid_idx]
    embeddings=embeddings[valid_idx,:]

    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, dataset.num_classes).long().to(logits.device)
    inserted_value = torch.zeros((1,)).long().to(logits.device)
    for ign_label in dataset.ignored_labels:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
    # print(valid_labels.shape,torch.min(valid_labels),torch.max(valid_labels))
    cls_loss = criterion['nll'](valid_logits, valid_labels).mean()
    end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels
    end_points['cls_loss'] = cls_loss

    # embeddings=embeddings[0:1000,:]
    # valid_labels=valid_labels[0:1000]

##########################################################
    if selected_num != 0:
        score = valid_logits.max(dim=1)[1]
        mask = score==valid_labels
        embeddings = embeddings[mask,:]
        valid_labels = valid_labels[mask]
        N, _ = embeddings.shape
        # if 0:
        #     props = torch.zeros((N))
        #     for i in range(19):
        #         cls_idx = torch.where(valid_labels==i)
        #         props[cls_idx] = sample_weight[i]
        #     idx = torch.multinomial(props, selected_num)
        #     embeddings = embeddings[idx,:]
        #     valid_labels = valid_labels[idx]
        #     end_points['embeddings'] = embeddings.cpu()
        #     end_points['em_labels'] = valid_labels.cpu()
        # else:
        index = torch.LongTensor(random.sample(range(N), selected_num))
        index=index.to(logits.device)
        embeddings = torch.index_select(embeddings, 0, index)
        valid_labels=torch.index_select(valid_labels, 0, index)
        end_points['embeddings']=embeddings.cpu()
        end_points['em_labels']=valid_labels.cpu()

    return end_points
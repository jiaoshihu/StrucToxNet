#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import pickle
import torch.nn as nn
import numpy as np
from model import pepmodel
from configuration import config as cf
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from util import util_metric
from util import util_loss
from util import util_record
from preprocess import  data_processing
import random
from torch.cuda.amp import GradScaler, autocast


# In[2]:


Model_save_dir = 'Model_saved'
if not os.path.exists(Model_save_dir):
    os.makedirs(Model_save_dir)
temp_model = "temp_model.pkl"


# In[3]:


config = cf.get_train_config()
scaler = GradScaler()


# In[4]:


with open('./data/combined_train.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('./data/combined_test.pkl', 'rb') as f:
    test_data = pickle.load(f)


# In[5]:


train_loader, test_loader = data_processing.create_data_loaders(train_data, test_data,config)


# In[6]:


contras_criterion = util_loss.ContrastiveLoss(config)


# In[7]:


def train_one_fold(train_iter, valid_iter, test_iter, model, optimizer, criterion, fold, config):
    train_acc_record = []
    train_loss_record = []

    val_metric_record = []
    val_loss_record = []

    test_metric_record = []
    test_loss_record = []
    threshold = config.threshold


    for epoch in range(1, config.epochs+1):
        print('=' * 20, 'Fold_{fold} current epoch: {epoch}'.format(fold= fold, epoch = epoch), '=' * 20)
        model.train()

        train_epoch_loss = 0
        train_total_num = 0
        train_correct_num = 0
        step = 0

        for batch in train_iter:
            label_list = []
            output_list = []
            logits_list = []
            optimizer.zero_grad()

            batch = batch.to(config.device)
            labels = batch.label.to(config.device)
            with autocast():
                outputs = model.forward(batch)
                logits = model.get_logits(batch)


                output_list.append(outputs)
                logits_list.append(logits)
                label_list.append(labels)

                output_b = torch.cat(output_list, dim=0)
                logits_b = torch.cat(logits_list, dim=0)
                label_b = torch.cat(label_list, dim=0)
                label_b = label_b.view(-1)

                label_pair = []
                contras_len = len(output_b) // 2
                label1 = label_b[:contras_len]
                label2 = label_b[contras_len:contras_len * 2]

                output1 = output_b[:contras_len]
                output2 = output_b[contras_len:contras_len * 2]

                for i in range(contras_len):
                    xor_label = (label1[i].long() ^ label2[i].long())
                    label_pair.append(xor_label.unsqueeze(0))
                contras_label = torch.cat(label_pair)

                contras_loss = contras_criterion(output1, output2, contras_label)
                ce_loss = util_loss.get_loss(logits_b, label_b, criterion)

                loss = ce_loss + contras_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            pred_prob_sort = torch.max(outputs, 1)
            pred_class = pred_prob_sort[1]

            for i in range(len(labels)):
                if labels[i].item() == pred_class[i].item():
                    train_correct_num +=1

            the_batch_size = labels.shape[0]
            train_epoch_loss += loss.item()
            train_total_num += the_batch_size

            step +=1

        train_acc_record.append(np.around((train_correct_num/train_total_num*100),decimals=3))
        train_loss_record.append(np.around(train_epoch_loss/step,decimals=3))

        valid_metric, valid_loss = model_eval(valid_iter, model,criterion,config)
        val_metric_record.append(valid_metric.cpu().detach().numpy())
        val_loss_record.append(np.around(valid_loss,decimals=3))

        test_metric, test_loss = model_eval(test_iter,model,criterion,config)
        test_metric_record.append(test_metric.cpu().detach().numpy())
        test_loss_record.append(np.around(test_loss,decimals=3))

        print("Validation Acc: ", valid_metric[0].item())
        print("Testing Acc: ", test_metric[0].item())

        if valid_metric[0].item() > threshold and  test_metric[0].item() > threshold:
            temp_best_model = 'fold_{fold}_epoch_{epoch}_model.pkl'.format(fold= fold, epoch = epoch)
            torch.save(model.state_dict(), os.path.join(Model_save_dir, temp_best_model))

#         lr_scheduler.step(valid_loss)

    return train_acc_record, val_metric_record, test_metric_record, \
           train_loss_record, val_loss_record, test_loss_record


# In[8]:


def model_eval(data_iter, model, criterion, config):

    label_pred = torch.empty([0], device=config.device)
    label_true = torch.empty([0], device=config.device)
    pred_prob = torch.empty([0], device=config.device)

    val_epoch_loss = 0
    step = 0

    model.eval()
    with torch.no_grad():
        for batch in data_iter:
            
            batch = batch.to(config.device)
            labels = batch.label.to(config.device)

            outputs = model.get_logits(batch)
            loss = util_loss.get_loss(outputs, labels, criterion)
            val_epoch_loss += loss.item()

            pred_prob_all = F.softmax(outputs, dim=1)
            pred_prob_positive = pred_prob_all[:, 1]
            pred_prob_sort = torch.max(outputs, 1)
            pred_class = pred_prob_sort[1]

            label_pred = torch.cat([label_pred, pred_class])
            label_true = torch.cat([label_true, labels])
            pred_prob = torch.cat([pred_prob, pred_prob_positive])

            step +=1

        metric, roc_data, prc_data = util_metric.caculate_metric(label_pred, label_true, pred_prob)
        val_avg_loss = float(val_epoch_loss/step)

    return metric, val_avg_loss


# In[ ]:


fold=1


model = pepmodel.EGNNModel(config).to(config.device)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), config.lr, weight_decay=config.wd)
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=10, min_lr=1e-6)

train_acc, val_metric, test_metric, train_loss, val_loss, test_loss = train_one_fold(train_loader, test_loader, test_loader, model, optimizer, criterion, fold+1, config)
util_record.metric_record(train_acc, val_metric, test_metric,fold)
util_record.loss_record(train_loss, val_loss, test_loss, fold)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





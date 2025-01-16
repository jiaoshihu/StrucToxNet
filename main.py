#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pickle
import pandas as pd
import numpy as np
from model import pepmodel
from configuration import config as cf
import torch.nn.functional as F
from util import util_metric
from preprocess import  data_processing



config = cf.get_train_config()


with open('./data/combined_test.pkl', 'rb') as f:
    test_data = pickle.load(f)

def read_fasta_file(filepath):
    with open(filepath, "r") as file:
        lines = [line.strip() for line in file]
        sequences_dict = {lines[i][1:]: lines[i+1] for i in range(0, len(lines), 2)}
    return sequences_dict



train_loader, test_loader = data_processing.create_data_loaders(test_data,config)


def model_eval(data_iter, model, config):
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

            pred_prob_all = F.softmax(outputs, dim=1)
            pred_prob_positive = pred_prob_all[:, 1]
            pred_prob_sort = torch.max(outputs, 1)
            pred_class = pred_prob_sort[1]

            label_pred = torch.cat([label_pred, pred_class])
            label_true = torch.cat([label_true, labels])
            pred_prob = torch.cat([pred_prob, pred_prob_positive])

            step += 1

        metric, roc_data, prc_data = util_metric.caculate_metric(label_pred, label_true, pred_prob)

    return metric, roc_data, prc_data, label_true.cpu().numpy(), pred_prob.cpu().numpy()


model = pepmodel.EGNNModel(config).to(config.device)
model.load_state_dict(torch.load('model/final_model.pkl', map_location=torch.device('cpu')))

valid_metric, roc_data, prc_data, true_label, probability = model_eval(test_loader, model, config)

roc_data_df = pd.DataFrame({
    'False Positive Rate': roc_data[0],
    'True Positive Rate': roc_data[1], })

roc_csv_path = f'./result/roc_test.csv'
roc_data_df.to_csv(roc_csv_path, index=False)

prc_data_save = {'recall': prc_data[0], 'precision': prc_data[1], 'AP': prc_data[2]}
np.save(f'./result/prc_data.npy', prc_data_save)

df_proba = pd.DataFrame({
    'true_label': true_label,
    'probability': probability})

proba_csv_path = f'./result/prolabel_probability.csv'
df_proba.to_csv(proba_csv_path, index=False)




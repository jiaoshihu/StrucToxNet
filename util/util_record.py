import csv
import pandas as pd
import os

def metric_record(train_acc, val_metric, test_metric, fold):
    train_acc = pd.DataFrame(train_acc)
    val_metric = pd.DataFrame(val_metric)
    test_metric = pd.DataFrame(test_metric)
    df_record = pd.concat([train_acc,val_metric,test_metric],axis=1,ignore_index=True)
    df_record.columns = ['Train_Acc','Acc (%)','AUC','Sn (%)','Sp (%)','MCC','F1','Precision','Acc (%)','AUC','Sn (%)','Sp (%)','MCC','F1','Precision']
    df_record.to_excel("./result/Metrics_{}_fold.xlsx".format(fold+1),index=True)


def loss_record(train_loss, val_loss, test_loss, fold):
    train_loss = [float(val) for val in train_loss]
    val_loss = [float(val) for val in val_loss]
    test_loss = [float(val) for val in test_loss]
    loss_record_file = "./result/Loss_{}_fold.xlsx".format(fold + 1)
    os.makedirs(os.path.dirname(loss_record_file), exist_ok=True)
    df_loss = pd.DataFrame({
        'epoch': range(1, len(train_loss) + 1),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss
    })
    df_loss.to_excel(loss_record_file, index=False, engine='openpyxl')
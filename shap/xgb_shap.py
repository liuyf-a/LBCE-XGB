#!/opt/share/bin/anaconda3/bin python
# coding: utf-8
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use("Agg")
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve,auc
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import joblib
import sys


#first read the data
def readtxt(filename):
    label=[]
    feature = []
    with open(filename,'r') as f:
        for line in f:
            line = line.split(',')
            feature.append([float(x) for x in line[1:769]])
            label.append([float(x) for x in line[0:1]])
    return np.array(feature),np.array(label)

def pr_auc_score(label,prob):
    precision, recall, _thresholds = precision_recall_curve(label, prob)
    area = auc(recall, precision)
    return area

aucprc = make_scorer(pr_auc_score)

train_file = sys.argv[1]
test_file = sys.argv[2]
x_train, y_train = readtxt(train_file)
x_test, y_test = readtxt(test_file)

xlf = xgb.XGBClassifier(max_depth=16,
                        learning_rate=0.01,
                        n_estimators=2000,
                        objective='binary:logistic',
                        nthread=-1,
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=1440,
                       missing=None)

xlf.fit(x_train, y_train)

explainer = shap.TreeExplainer(xlf)
shap_values = explainer.shap_values(x_train)
xlf_shap_import = np.mean(abs(shap_values),axis=0) #this is the shap importance for each feature based on all train data
print(xlf_shap_import)
#sort the shap importance
idx_sorted = np.argsort(xlf_shap_import) #this is ascend
idx_sorted = idx_sorted[::-1]

joblib.dump(idx_sorted, './B-cell/blind387/CLS.np')

train_fea = np.array(x_train[:,idx_sorted[:]])
test_fea = np.array(x_test[:,idx_sorted[:]])
# 将排序后的特征写入新文件中
'''
new_train_file = open('./B-cell/tr_CLS.txt', 'w')
#print(x_train[:,idx_sorted[:100]])
for j in range(len(train_fea)):
    new_train_file.write(str(int(y_train[j][0])))
    for i in range(len(train_fea[j])):
        new_train_file.write(',')
        new_train_file.write(str(train_fea[j][i]))
    new_train_file.write('\n')

new_test_file = open('./B-cell/te_CLS.txt', 'w')
#print(x_train[:,idx_sorted[:100]])
for j in range(len(test_fea)):
    new_test_file.write(str(int(y_test[j][0])))
    for i in range(len(test_fea[j])):
        new_test_file.write(',')
        new_test_file.write(str(test_fea[j][i]))
    new_test_file.write('\n')
'''
#idx_sorted_ds = np.argsort(-xlf_shap_import) #this is descend
param_grid = {
    'max_depth': [12, 16, 20],
    'learning_rate': [0.001, 0.01, 0.05],
    'n_estimators': [1400, 1800, 2200],
    }
aucroc_on_train = []
aucroc_on_test = []
featn = []
auc_value = []
#for i in range(5,len(idx_sorted),5):
for i in range(5, 201, 5):
    X_train_tmp = x_train[:,idx_sorted[:i]]
    X_test_tmp = x_test[:,idx_sorted[:i]]
    xlf_tmp = xgb.XGBClassifier(max_depth=10,
                        learning_rate=0.01,
                        n_estimators=2000,
                        objective='binary:logistic',
                        nthread=-1,
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=1440,
                       missing=None)
    optimized_GBM = GridSearchCV(xlf_tmp, param_grid=param_grid, scoring='roc_auc', cv=5, refit=True,verbose=1, return_train_score=True,n_jobs=-1)
    optimized_GBM.fit(X_train_tmp, y_train)
    print("Best parameters:{}".format(optimized_GBM.best_params_))
    print("Test set roc_auc:{:.3f}".format(optimized_GBM.score(X_test_tmp,y_test)))
    print("Best roc_auc on train set:{:.3f}".format(optimized_GBM.best_score_))
    #featn.append(len(idx_sorted)-i)
    featn.append(i)
    aucroc_on_train.append(optimized_GBM.best_score_)
    aucroc_on_test.append(optimized_GBM.score(X_test_tmp,y_test))

newdf = pd.DataFrame({'featn':featn,'aucroc_on_train':aucroc_on_train,'aucroc_on_test':aucroc_on_test})
newdf.to_csv('./B-cell/CLS.csv')

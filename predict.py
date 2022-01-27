import joblib
from sklearn.calibration import CalibratedClassifierCV as cc, calibration_curve
from Bio import SeqIO
from pydpi.pypro import PyPro
import sys
import numpy as np
import pandas as pd
import warnings
import os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
from sklearn import svm, datasets, metrics
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, precision_recall_curve, precision_recall_fscore_support
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_score, recall_score, roc_auc_score, auc, matthews_corrcoef, classification_report
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold, LeaveOneOut, train_test_split
from sklearn.decomposition import PCA, TruncatedSVD as svd
from scipy import interp
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, SelectFromModel
#from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import make_scorer
import argparse
import pickle, gzip
import math




protein = PyPro()


def readAAP(file):  # read AAP features from the AAP textfile
    try:
        aapdic = {}
        aapdata = open(file, 'r')
        for l in aapdata.readlines():
            aapdic[l.split()[0]] = float(l.split()[1])
        aapdata.close()
        return aapdic
    except:
        print("Error in reading AAP feature file. Please make sure that the AAP file is correctly formatted")
        sys.exit()


def readAAT(file):  # read AAT features from the AAT textfile
    try:
        aatdic = {}
        aatdata = open(file, 'r')
        for l in aatdata.readlines():
            aatdic[l.split()[0][0:3]] = float(l.split()[1])
        aatdata.close()
        return aatdic
    except:
        print("Error in reading AAT feature file. Please make sure that the AAT file is correctly formatted")
        sys.exit()


def aap(pep, aapdic, avg):  # return AAP features for the peptides
    feature = []
    for a in pep:
        # print(a)
        if int(avg) == 0:
            score = []
            count = 0
            for i in range(0, len(a) - 1):
                try:
                    score.append(round(float(aapdic[a[i:i + 2]]), 4))
                    # score += float(aapdic[a[i:i + 3]])
                    count += 1
                except KeyError:
                    # print(a[i:i + 3])
                    score.append(float(-1))
                    # score += -1
                    count += 1
                    continue
            # averagescore = score / count
            feature.append(score)
        if int(avg) == 1:
            score = 0
            count = 0
            for i in range(0, len(a) - 1):
                try:
                    score += float(aapdic[a[i:i + 2]])
                    count += 1
                except KeyError:
                    score += -1
                    count += 1
                    continue
            if count != 0:
                averagescore = score / count
            else:
                averagescore = 0
            feature.append(round(float(averagescore), 4))
    return feature


def aat(pep, aatdic, avg):  # return AAT features for the peptides
    feature = []
    for a in pep:
        if int(avg) == 0:
            # print(a)
            score = []
            count = 0
            for i in range(0, len(a) - 2):
                try:
                    score.append(round(float(aatdic[a[i:i + 3]]), 4))
                    # score += float(aapdic[a[i:i + 3]])
                    count += 1
                except KeyError:
                    # print(a[i:i + 3])
                    score.append(float(-1))
                    # score += -1
                    count += 1
                    continue
            # averagescore = score / count
            feature.append(score)
        if int(avg) == 1:
            score = 0
            count = 0
            for i in range(0, len(a) - 2):
                try:
                    score += float(aatdic[a[i:i + 3]])
                    count += 1
                except KeyError:
                    score += -1
                    count += 1
                    continue
            # print(a, score)
            if count != 0:
                averagescore = score / count
            else:
                averagescore = 0
            feature.append(round(float(averagescore), 4))
    return feature


def AAC(pep):  # Single Amino Acid Composition feature
    feature = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        aac = protein.GetAAComp()
        feature.append(list(aac.values()))
        name = list(aac.keys())
    return feature, name

def DPC(pep):  # Dipeptide Composition feature
    feature = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        dpc = protein.GetDPComp()
        feature.append(list(dpc.values()))
        name = list(dpc.keys())
    return feature, name

def PAAC(pep):
    feature = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        #paac=protein.GetMoranAuto()
        paac = protein.GetPAAC(lamda=4)
        feature.append(list(paac.values()))
        name = list(paac.keys())
    return feature, name


def bertfea(file):
    feature = []
    import joblib
    idx_sorted = joblib.load('./shap/CLS.np')
    x_test = np.array(pd.read_csv(file,header=None,index_col=None,usecols=[i for i in range(1,769)]))
    return x_test[:, idx_sorted[:130]]


def readpeptides(posfile, negfile):  # return the peptides from input peptide list file
    posdata = open(posfile, 'r')
    pos = []
    for l in posdata.readlines():
        if l[0] == '>':
            continue
        else:
            pos.append(l.strip('\t0\n'))
    posdata.close()
    negdata = open(negfile, 'r')
    neg = []
    for l in negdata.readlines():
        if l[0] == '>':
            continue
        else:
            neg.append(l.strip('\t0\n'))
    negdata.close()
    #print(pos)
    return pos, neg


def combinefeature(pep, featurelist, dataset):
    a=np.empty([len(pep), 1])
    fname=[]
    scaling = StandardScaler()
    #pca = svd(n_components=300)
    pca = PCA(0.99)
    vocab_name = []
    #pca = PCA(n_components=10)
    #print(a)
    if 'aap' in featurelist:
        aapdic = readAAP("./models/aap-general.txt.normal")
        f_aap = np.array([aap(pep, aapdic, 1)]).T
        a = np.column_stack((a,f_aap))
        #a = scaling.fit_transform(a)
        fname.append('AAP')
        #print(f_aap)
    if 'aat' in featurelist:
        aatdic = readAAT("./models/aat-general.txt.normal")
        f_aat = np.array([aat(pep, aatdic, 1)]).T
        a = np.column_stack((a, f_aat))
        #a = scaling.fit_transform(a)
        fname.append('AAT')
        #print(f_aat)

    if 'aac' in featurelist:
        f_aac, name = AAC(pep)
        a = np.column_stack((a, np.array(f_aac)))
        fname = fname + name
   
    if 'dpc' in featurelist:
        f_dpc, name = DPC(pep)
        # f_dpc = np.average(f_dpc, axis =1)
        a = np.column_stack((a, np.array(f_dpc)))
        fname = fname + name
 
    if 'paac' in featurelist:
        f_paac, name = PAAC(pep)
        #f_paac = pca.fit_transform(f_paac)
        a = np.column_stack((a, np.array(f_paac)))
        fname = fname + name
   
        
    if 'bertfea' in featurelist:
        f_bertfea = np.array(bertfea('./datasets/BERT_feature/independent/CLS_fea.txt'))
        a = np.column_stack((a, f_bertfea))
        fname = fname + ['bertfea']*len(f_bertfea)       

    return a[:,1:], fname, vocab_name

def precision_0(y_true, y_pred, labels=None, average='binary', sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate prec for neg class
    '''
    p, _, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=0,
                                                 average=average,
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return p

def recall_0(y_true, y_pred, labels=None, average='binary', sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate recall for neg class
    '''
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=0,
                                                 average=average,
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return r

def f1_0(y_true, y_pred, labels=None, average='binary', sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate f1 for neg class
    '''
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=0,
                                                 average=average,
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return f

def run_training(pos, neg, dataset):
    pep_combined = pos + neg
    # aap aat dpc aac paac qso ctd
    featurelist = ['aac', 'aap', 'aat', 'bertfea']
    print(featurelist)
    features, fname, vocab = combinefeature(pep_combined, featurelist, dataset) # 'aap', 'aat', 'aac'
    print(len(features[0]))
    '''for i in range(len(features)):
    	print(features[i])'''
    target = [1] * len(pos) + [0] * len(neg)
    train(features, target)

def train(features, target):
    with open('./models/model.pickle', 'rb') as fin:
        alldata = pickle.load(fin)
    print(alldata.keys())
    model1 = alldata['model']
    f_scaling = alldata['scaling']
    #f_scaling.fit(features)
    x_test = f_scaling.transform(features)
    y = np.array(target)
    y_pred = model1.predict(x_test)
    y_scores = model1.predict_proba(x_test)[:, 1]


    TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
    print('TN, FP, FN, TP:', TN, FP, FN, TP)

    Specificity = TN / (TN + FP)
    ACC = (TP + TN) / (TP + FP + FN + TN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1Score = 2 * TP / (2 * TP + FP + FN)
    MCC = float(TP * TN - FP * FN) / math.sqrt(float(TP + FP) * float(TP + FN) * float(TN + FP) * float(TN + FN))

    p, r, thresh = metrics.precision_recall_curve(y, y_scores)
    pr_auc = metrics.auc(r, p)

    ro_auc = metrics.roc_auc_score(y, y_scores)

    print('Specificity:', Specificity, 'ACC:', ACC, 'Precision:', Precision, 'Recall/Sensitive:', Recall,
          'F1Score:', F1Score, 'MCC:', MCC, 'auprc:', pr_auc, 'auroc:', ro_auc)

    '''MRF = RFClassifier(x, y)
    MRF.tune_and_eval("4mer_rf")'''


if __name__ == "__main__":
    dataset = sys.argv[1]
    pos, neg = readpeptides("./datasets/training/"+dataset+"/pos.txt",
                            "./datasets/training/"+dataset+"/neg.txt")
    #print(pos, neg)
    run_training(pos, neg, dataset)

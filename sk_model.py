import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.externals import joblib


Fmd = './data/dataset.csv'
Fpr = './data/dataset_pre.csv'
def load_dataframe():
    df_md = pd.read_csv(Fmd)
    df_pr = pd.read_csv(Fpr)
    return (df_md, df_pr)

def get_pre_dataset(pre_type=1):
    df_pr = pd.read_csv(Fpr)
    df = df_pr[df_pr['pre_type']==pre_type]
    fea_lst = list(df.columns)
    fea_lst.remove('uid')
    fea_lst.remove('pre_type')
    #fea_lst.remove('tag')
    return (df, fea_lst)

def get_pre_list(clf, df, fl):
    prob = clf.predict_proba(df[fl])
    uids = list(df['uid'])
    rs = [(uids[i],prob[i][1]) for i in range(0,len(uids))]
    rs.sort(key=lambda x:x[1], reverse=True)
    return rs

def get_dataframe(pre_type=1, split=True):
    df_md = pd.read_csv(Fmd)
    df = df_md[df_md['pre_type']==pre_type]
    if split:
        train_set, test_set = train_test_split(df, test_size=0.1)
    else:
        train_set = df
        test_set = None
    fea_lst = list(train_set.columns)
    fea_lst.remove('uid')
    fea_lst.remove('pre_type')
    fea_lst.remove('tag')
    return (train_set, test_set, fea_lst)

def test_BernoulliNB():
    clf = BernoulliNB()
    for ptype in range(1,8):
        train_set, test_set, fea_lst = get_dataframe(ptype)
        clf.fit(train_set[fea_lst], train_set['tag'])
        s = clf.score(test_set[fea_lst], test_set['tag'])
        print('ptype:',ptype,' score:',s)

def test_Ada():
    clf = Ada(n_estimators=200)
    for ptype in range(1,8):
        train_set, test_set, fea_lst = get_dataframe(ptype)
        clf.fit(train_set[fea_lst], train_set['tag'])
        s = clf.score(test_set[fea_lst], test_set['tag'])
        joblib.dump(clf,'clf_{}'.format(ptype))
        print('ptype:',ptype,' score:',s)

def All_Ada():
    clf = Ada(n_estimators=200)
    for ptype in range(1,8):
        print('training {}'.format(ptype))
        train_set, test_set, fea_lst = get_dataframe(ptype, False)
        clf.fit(train_set[fea_lst], train_set['tag'])
        joblib.dump(clf,'clf_all_{}'.format(ptype))

def my_pre():
    pre_set, fl = get_pre_dataset()
    labels = {}
    uids = pre_set['uid']
    joblib.dump(uids, 'uids.pkl')
    for ptype in range(1, 8):
        clf = joblib.load('clf_all_{}'.format(ptype))
        labels[ptype] = clf.predict(pre_set[fl])
    joblib.dump(labels, 'labs.pkl')

def final_uids():
    uids = list(joblib.load('uids.pkl'))
    labs = joblib.load('labs.pkl')
    rs = {}
    for i in range(1,8):
        for u, l in zip(uids,labs[i]):
            if l == 1:
                rs.setdefault(i,[]).append(u)
    joblib.dump(rs, 'pre_uids.pkl')
   

if __name__ == '__main__':
    #All_Ada()
    #my_pre()
    final_uids()

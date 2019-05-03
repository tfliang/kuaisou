import lightgbm as lgb
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pickle as pk


Fmd = './data/dataset.csv'
Fpr = './data/dataset_pre.csv'

def get_pre_dataset(pre_type=1):
    df_pr = pd.read_csv(Fpr)
    df = df_pr[df_pr['pre_type']==pre_type]
    fea_lst = list(df.columns)
    fea_lst.remove('uid')
    fea_lst.remove('pre_type')
    #fea_lst.remove('tag')
    return (df, fea_lst)

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

def get_score(clf, test_set, fl):
    prod = list(clf.predict(test_set[fl]))
    pre = [(prod[i], test_set['tag'].iloc[i], test_set['uid'].iloc[i]) for i in range(0,len(prod))]
    pre.sort(key=lambda x:x[0], reverse=True)

    u11 = len([1 for tmp in pre if tmp[1]==1 and tmp[0]>=0.5])
    u10 = len([1 for tmp in pre if tmp[1]==1 and tmp[0]<0.5])
    u01 = len([1 for tmp in pre if tmp[1]==0 and tmp[0]>=0.5])
    u00 = len([1 for tmp in pre if tmp[1]==0 and tmp[0]<0.5])
    wrong_pre = [
        tmp for tmp in pre
        if (tmp[1]==1 and tmp[0]<0.5) or (tmp[1]==0 and tmp[0]>=0.5)
    ]
    u1 = u11 + u10
    u1_pre = len([1 for tmp in pre[:u1] if tmp[1]==1])
    wrong_pre2 = [tmp for tmp in pre[:u1] if tmp[1]==0]
    #print(wrong_pre)
    F_score = u1_pre/u1
    print(F_score)
    print(u11, u10, u00, u01, len(pre))

def test():
    train_set, test_set, fl = get_dataframe()
    tr = lgb.Dataset(train_set[fl], label=train_set['tag']) 
    params = dict(
        num_leaves=30,
        num_trees=100,
        objective='binary'
    )
    num_round = 100
    clf = lgb.train(params, tr, num_round)
    get_score(clf, test_set, fl)
    #print(get_score(clf, test_set, fl))
    #clf = lgb.train(params, tr, num_round)
    #print(get_score(clf, test_set, fl))

def sub_uids():
    pre_df, fl = get_pre_dataset()
    train_set, tmp1, tmp2 = get_dataframe(split=False)
    tr = lgb.Dataset(train_set[fl], label=train_set['tag']) 
    params = dict(
        num_leaves=30,
        num_trees=100,
        objective='binary'
    )
    num_round = 100
    clf = lgb.train(params, tr, num_round)
    pre = list(clf.predict(pre_df[fl]))
    uids = list(pre_df['uid'])
    rday = list(pre_df['reg_day'])
    pre_uids = [(uids[i],rday[i],pre[i]) for i in range(0,len(pre))]
    pre_uids.sort(key=lambda x:x[2],reverse=True)
    pk.dump(pre_uids,open('./data/lgb_uids.pkl','wb'))
    return
    
    with open('uids_glb_2000.csv','w') as f:
        for uid in pre_uids[0:23727+2000]:
            f.write(str(uid[0])+'\n')


if __name__ == '__main__':
    #sub_uids()
    test()

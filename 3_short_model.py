import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle as pk
import json
from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.model_selection import train_test_split


fpath = './data/'
f_json = fpath+'2_pre_data.json'
f_pkl = fpath+'2_pre_data.pkl'
f_json_d1 = fpath+'2_pre_data_d1.json'
f_pkl_d1 = fpath+'2_pre_data_d1.pkl'


def delete_device_1():
    rs = load_data(1)
    rs_new = {}
    with open(f_json_d1,'w') as f:
        for uid, data in rs.items():
            if data['userdata'][2]!=1:
                rs_new[uid] = data
                f.write(json.dumps({uid:data})+'\n')
    pk.dump(rs_new,open(f_pkl_d1,'wb'))
    print(len(rs_new))

def yield_load_json(full=0):
    if full:
        with open(f_json) as f:
            for myline in f:
                myl = json.loads(myline)
                uid = list(myl.keys())[0]
                yield (uid, myl[uid])
    else:
        with open(f_json_d1) as f:
            for myline in f:
                myl = json.loads(myline)
                uid = list(myl.keys())[0]
                yield (uid, myl[uid])

def load_data(full=0):
    if full:
        rs = pk.load(open(f_pkl, 'rb'))
    else:
        rs = pk.load(open(f_pkl_d1, 'rb'))
    return rs

def get_feature_short(uid, data, m, n, pre=False):
    if pre:
        if (len(data['tag'])-data['tag'].index(1)) < m:
            return
        index_start = data['tag'].index(1)
        index_tag = index_start + m
        tmp_tag = data['tag'][:index_tag]
        tmp_video = data['video'][:index_tag]
        tmp_act_uniq = data['act_uniq'][:index_tag]
        tmp_be_act = data['be_act'][:index_tag]
        tmp_act_act = data['act_act'][:index_tag]
        tmp_act_page = data['act_page'][:index_tag]
        
    else:

        if (len(data['tag'])-data['tag'].index(1)) < (m+n):
            return
        index_start = data['tag'].index(1)
        index_tag = index_start + m
        tmp_tag = data['tag'][:index_tag]
        tmp_video = data['video'][:index_tag]
        tmp_act_uniq = data['act_uniq'][:index_tag]
        tmp_be_act = data['be_act'][:index_tag]
        tmp_act_act = data['act_act'][:index_tag]
        tmp_act_page = data['act_page'][:index_tag]

    feature = []
    fnames = []
    # user data
    user_source = data['userdata'][1]
    user_device = data['userdata'][2]
    reg_day = index_start + 1
    #feature.extend([user_source, user_device, reg_day % 7])
    feature.extend([str(user_source), str(user_device), str(reg_day)])
    fnames.extend(['user_source', 'user_device', 'reg_day'])
    # general feature
    log_length = len(tmp_tag) - tmp_tag.index(1)
    tag_sum = sum(tmp_tag)
    vid_sum = sum(tmp_video)
    act_sum = sum(tmp_act_uniq)
    acted_sum = sum(map(sum, tmp_be_act))
    act_avg1 = act_sum/log_length
    act_avg2 = act_sum/tag_sum

    '''
    feature.extend([
        vid_sum, act_sum, acted_sum  
        ])
    fnames.extend([
        'vid_sum', 'act_sum', 'acted_sum'  
        ])

    '''
    feature.extend([
        log_length, tag_sum, vid_sum, act_sum, acted_sum,  
        act_avg1, act_avg2
        ])
    fnames.extend([
        'log_length', 'tag_sum', 'vid_sum', 'act_sum', 'acted_sum',  
        'act_avg1', 'act_avg2'
        ])

    act_type = list(sum(np.array(tmp_act_act)))
    acted_type = list(sum(np.array(tmp_be_act)))
    page_type = list(sum(np.array(tmp_act_page)))
    types = act_type + acted_type + page_type
    feature.extend(types)
    fnames.extend([
        'act_0','act_1','act_2','act_3','act_4','act_5',
        'acted_0','acted_1','acted_2','acted_3','acted_4','acted_5',
        'page_0','page_1','page_2','page_3','page_4'
        ])

    # sequence feature
    '''
    act_latest = tmp_act_uniq[-1] 
    act_latest2 = tmp_act_uniq[-2] 
    video_latest = tmp_video[-1]
    feature.extend([act_latest, act_latest2, video_latest])
    fnames.extend(['act_latest', 'act_latest2', 'video_latest'])
    '''
    '''
    ntail = 5
    act_tail = tmp_act_uniq[index_tag-ntail:index_tag] 
    feature.extend(act_tail)
    fnames.extend(['act_tail_{}'.format(i) for i in range(1,ntail+1)])
    '''

    act_all = tmp_act_uniq[index_start:index_tag] 
    feature.extend(act_all)
    fnames.extend(['act_day_{}'.format(i) for i in range(1,m+1)])

    '''
    video_all = tmp_video[index_start:index_tag] 
    feature.extend(video_all)
    fnames.extend(['video_day_{}'.format(i) for i in range(1,m+1)])
    '''

    if pre:
        # uid, label
        feature.extend([uid])
        fnames.extend(['uid'])
    else:
        # uid, label
        tmp_label = 1 if sum(data['tag'][index_tag:index_tag+n])>0 else 0
        #tmp_label = sum(data['tag'][index_tag:index_tag+n])
        feature.extend([uid, tmp_label])
        fnames.extend(['uid', 'label'])
    # return
    return (fnames, feature)

def split_data_short(rs, params, day=21, m=5, n=5, num_round=100):
    #print('loading data')
    #rs = load_data()
    print('getting feature')
    feature_data = []
    feature_train = []
    feature_test = []

    for uid, data in rs.items():
        if data['userdata'][0] <= day:
            fname, feature = get_feature_short(uid,data,m,n)
            feature_data.append(feature)
            
    df = pd.DataFrame(data=feature_data, columns=fname)
    df['user_source'] = df['user_source'].astype('category')
    df['user_device'] = df['user_device'].astype('category')
    df['reg_day'] = df['reg_day'].astype('category')
    '''
    '''

    df_train, df_test = train_test_split(df, test_size=0.1)
    '''
    for uid, data in rs.items():
        if data['userdata'][0] == day:
            fname, feature = get_feature_short(uid,data,m,n)
            feature_test.append(feature)
        elif data['userdata'][0] < day:
            fname, feature = get_feature_short(uid,data,m,n)
            feature_train.append(feature)
    print(fname)
    df_train = pd.DataFrame(data=feature_train, columns=fname)
    df_test = pd.DataFrame(data=feature_test, columns=fname)
    '''

    train_set = lgb.Dataset(df_train[fname[:-2]], label=df_train['label'])
    print('training')
    clf = lgb.train(params, train_set, 100)
    ifi = clf.feature_importance()
    print([(fname[i], ifi[i]) for i in range(0, len(ifi))])
    prod = list(clf.predict(df_test[fname[:-2]]))
    tmp_rs = [(df_test['uid'].iloc[i],df_test['label'].iloc[i],prod[i]) for i in range(0,len(prod))]
    tmp_rs.sort(key=lambda x:x[2], reverse=True)
    uids_11 = [tmp for tmp in tmp_rs if (tmp[1]>0) and (tmp[2]>=0.5)]
    uids_10 = [tmp for tmp in tmp_rs if (tmp[1]>0) and (tmp[2]<0.5)]
    uids_00 = [tmp for tmp in tmp_rs if (tmp[1]==0) and (tmp[2]<0.5)]
    uids_01 = [tmp for tmp in tmp_rs if (tmp[1]==0) and (tmp[2]>=0.5)]
    n1 = len(uids_11)+len(uids_10)
    Fo = [ tmp for tmp in tmp_rs[:n1] if tmp[1]>0]
    Fx = [ tmp for tmp in tmp_rs[:n1] if tmp[1]==0]
    print(len(prod),len(uids_11),len(uids_10),len(uids_00),len(uids_01))
    print(len(uids_11+uids_00)/len(prod),len(uids_11)/len(uids_11+uids_10),len(uids_00)/len(uids_00+uids_01))
    tmp_F = len(Fo)/n1
    tmp_R = n1/len(prod)
    tmp_G = (tmp_F - tmp_R) / tmp_R
    print('F',tmp_F,'G',tmp_G,'R',tmp_R)
    #print(Fx)
    #print(uids_10)
    #print(uids_01)
    #return tmp_F-tmp_R
    return tmp_F
    
def short_model(m, n):
    print('short model ================', m, n)
    rs = load_data()
    params = dict(
        num_leaves=30,
        num_trees=250,
        objective='binary'
    )
    day = 31-n-m

    print('getting feature')
    feature_data = []
    for uid, data in rs.items():
        if data['userdata'][0] <= day:
            fname, feature = get_feature_short(uid,data,m,n)
            feature_data.append(feature)
            '''
            gg = get_feature_long2(uid,data,m,n)
            #gg = get_feature_long(uid,data,m,n)
            for fname, feature in gg:
                feature_data.append(feature)
            '''
    df_train = pd.DataFrame(data=feature_data, columns=fname)
    df_train['user_source'] = df_train['user_source'].astype('category')
    df_train['user_device'] = df_train['user_device'].astype('category')
    df_train['reg_day'] = df_train['reg_day'].astype('category')

    feature_pre = []
    for uid, data in rs.items():
        if data['userdata'][0] <= 31-m:
            fname, feature = get_feature_short(uid,data,m,n,pre=True)
            feature_pre.append(feature)
            '''
            gg = get_feature_short(uid,data,m,n,pre=True)
            #gg = get_feature_long2(uid,data,m,n,pre=True)
            #gg = get_feature_long(uid,data,m,n)
            for fname, feature in gg:
                feature_pre.append(feature)
            '''
    df_pre = pd.DataFrame(data=feature_pre, columns=fname)
    df_pre['user_source'] = df_pre['user_source'].astype('category')
    df_pre['user_device'] = df_pre['user_device'].astype('category')
    df_pre['reg_day'] = df_pre['reg_day'].astype('category')
    #df_train, df_test = train_test_split(df, test_size=0.2)
    train_set = lgb.Dataset(df_train[fname[:-2]], label=df_train['label'])
    num_round = 100
    print('training')
    clf = lgb.train(params, train_set, 100)
    ifi = clf.feature_importance()
    print([(fname[i], ifi[i]) for i in range(0, len(ifi))])

    prod = list(clf.predict(df_pre[fname[:-1]]))
    tmp_rs = [(df_pre['uid'].iloc[i],df_pre['reg_day'].iloc[i],prod[i]) for i in range(0,len(prod))]
    tmp_rs.sort(key=lambda x:x[2], reverse=True)
    print('saving')
    pk.dump(tmp_rs, open('./data/3short_{}.pkl'.format(m),'wb'))

def test():
    rs = load_data()

    '''
    params = dict(
        num_leaves=30,
        num_trees=250
    )
    '''
    params = dict(
        num_leaves=30,
        num_trees=250,
        objective='binary'
    )
    m = 2
    n = 3
    all_s = []
    for m in range(2, 3):
        s=[]
        for a in range(1,4):
            day = 31-n-m
            params = dict(
                num_leaves=30,
                num_trees=250,
                objective='binary'
            )
            tmp = split_data_short(rs, params, day, m, n, 100)
            s.append(tmp)
        print('=============================================== short model',m,n)
        print(s,sum(s)/len(s))
        print('=============================================== short model',m,n)
        all_s.append((m,sum(s)/len(s)))
    print(all_s)
        

if __name__ == '__main__':
    short_model(1,3)
    #test()

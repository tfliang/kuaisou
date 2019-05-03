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

def get_feature_long2(uid, data, m, n, pre=False):
    index_start = data['tag'].index(1)
    if pre:
        if (len(data['tag'])-data['tag'].index(1)) < m:
            return
        index_tag_min = 30
        index_tag_max = 31
    else:
        if (len(data['tag'])-data['tag'].index(1)) < (m+n):
            return
        index_tag_min = index_start + m
        index_tag_max = 31 - n

    for index_tag in range(index_tag_min, index_tag_max):
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
        feature.extend([user_source, user_device, reg_day])
        fnames.extend(['user_source', 'user_device', 'reg_day'])
        # general feature
        log_length = len(tmp_tag) - tmp_tag.index(1)
        tag_sum = sum(tmp_tag)
        vid_sum = sum(tmp_video)
        act_sum = sum(tmp_act_uniq)
        acted_sum = sum(map(sum, tmp_be_act))
        act_avg1 = act_sum/log_length
        act_avg2 = act_sum/tag_sum
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
        ntail = 4
        nhead = 4
        if m >= 5:
            act_tail = tmp_act_uniq[index_tag-ntail:index_tag] 
            feature.extend(act_tail)
            fnames.extend(['act_tail_{}'.format(i) for i in range(1,ntail+1)])

            tag_tail = tmp_tag[index_tag-ntail:index_tag] 
            feature.extend(tag_tail)
            fnames.extend(['tag_tail_{}'.format(i) for i in range(1,ntail+1)])

            video_tail = tmp_video[index_tag-ntail:index_tag] 
            feature.extend(video_tail)
            fnames.extend(['video_tail_{}'.format(i) for i in range(1,ntail+1)])

            '''
            act_head = tmp_act_uniq[index_start:index_start+nhead] 
            feature.extend(act_head)
            fnames.extend(['act_head_{}'.format(i) for i in range(1,nhead+1)])
            video_head = tmp_video[index_start:index_start+nhead] 
            feature.extend(video_head)
            fnames.extend(['video_head_{}'.format(i) for i in range(1,nhead+1)])
            '''
        elif m >=2:
            act_t1 = tmp_act_uniq[-1] 
            act_t2 = tmp_act_uniq[-2] 
            video_t1 = tmp_video[-1]
            feature.extend([act_t1, act_t2, video_t1])
            fnames.extend(['act_t1', 'act_t2', 'video_t1'])
        else:
            act_t1 = tmp_act_uniq[-1] 
            video_t1 = tmp_video[-1]
            feature.extend([act_t1, video_t1])
            fnames.extend(['act_t1', 'video_t1'])
    
        tag_gaps = [len(i) for i in ''.join(map(str,tmp_tag[index_start:index_tag])).split('1')]
        tag_tags = [len(i) for i in ''.join(map(str,tmp_tag[index_start:index_tag])).split('0')]
        max_gaps = max(tag_gaps)
        latest_gap = tag_gaps[-1]
        max_tags = max(tag_tags)
        latest_tag = tag_tags[-1]
        feature.extend([max_gaps, latest_gap, max_tags, latest_tag])
        fnames.extend(['max_gaps', 'latest_gap', 'max_tags', 'latest_tag'])
        # uid, label
        if pre:
            feature.extend([uid])
            fnames.extend(['uid'])
        else:
            tmp_label = 1 if sum(data['tag'][index_tag:index_tag+n])>0 else 0
            #tmp_label = sum(data['tag'][index_tag:index_tag+n])
            feature.extend([uid, tmp_label])
            fnames.extend(['uid', 'label'])
        # return
        yield (fnames, feature)

def split_data_long2(rs, params, day, m, n):
    #print('loading data')
    #rs = load_data()
    print('getting feature')
    feature_data = []
    feature_train = []
    feature_test = []

    '''
    for uid, data in rs.items():
        if data['userdata'][0] <= day:
            gg = get_feature_long2(uid,data,m,n)
            #gg = get_feature_long(uid,data,m,n)
            for fname, feature in gg:
                feature_data.append(feature)
            
    df = pd.DataFrame(data=feature_data, columns=fname)
    df_train, df_test = train_test_split(df, test_size=0.2)
    '''
    for uid, data in rs.items():
        if data['userdata'][0] == day:
            gg = get_feature_long2(uid,data,m,n)
            for fname, feature in gg:
                feature_test.append(feature)
        elif data['userdata'][0] < day:
            gg = get_feature_long2(uid,data,m,n)
            for fname, feature in gg:
                feature_train.append(feature)
    df_train = pd.DataFrame(data=feature_train, columns=fname)
    #print(df_train['label'].min(),df_train['label'].max())
    df_test = pd.DataFrame(data=feature_test, columns=fname)

    train_set = lgb.Dataset(df_train[fname[:-2]], label=df_train['label'])
    num_round = 100
    print('training')
    clf = lgb.train(params, train_set, 100)
    clf2 = lgb.train(params, train_set, 100)
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
    print('F',len(Fo)/n1)

    print('------------------- second')
    print('training')
    prod = list(clf2.predict(df_test[fname[:-2]]))
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
    print('F',len(Fo)/n1)
    #print(Fx)
    #print(uids_10)
    #print(uids_01)
    #print(tmp_rs[:10])
    #print(tmp_rs[-10:])

def get_feature_long(uid, data, m, n):
    if (len(data['tag'])-data['tag'].index(1)) < (m+n):
        return
    index_start = data['tag'].index(1)
    for index_tag in range(index_start + m, 31 - n):
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
        feature.extend([user_source, user_device])
        fnames.extend(['user_source', 'user_device'])
        # general feature
        log_length = len(tmp_tag) - tmp_tag.index(1)
        tag_sum = sum(tmp_tag)
        vid_sum = sum(tmp_video)
        act_sum = sum(tmp_act_uniq)
        acted_sum = sum(map(sum, tmp_be_act))
        act_avg1 = act_sum/log_length
        act_avg2 = act_sum/tag_sum
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
        act_latest = tmp_act_uniq[-1] 
        video_latest = tmp_video[-1]
        feature.extend([act_latest, video_latest])
        fnames.extend(['act_latest', 'video_latest'])
        if m>1:
            act_latest2 = tmp_act_uniq[-2] 
            feature.append(act_latest2)
            fnames.append('act_latest2')

        '''
        act_all = tmp_act_uniq[index_start:index_tag] 
        feature.extend(act_all)
        fnames.extend(['act_day_{}'.format(i) for i in range(1,m+1)])
        video_all = tmp_video[index_start:index_tag] 
        feature.extend(video_all)
        fnames.extend(['video_day_{}'.format(i) for i in range(1,m+1)])
        '''
    
        tag_gaps = [len(i) for i in ''.join(map(str,tmp_tag[index_start:index_tag])).split('1')]
        tag_tags = [len(i) for i in ''.join(map(str,tmp_tag[index_start:index_tag])).split('0')]
        max_gaps = max(tag_gaps)
        latest_gap = tag_gaps[-1]
        max_tags = max(tag_tags)
        latest_tag = tag_tags[-1]
        feature.extend([max_gaps, latest_gap, max_tags, latest_tag])
        fnames.extend(['max_gaps', 'latest_gap', 'max_tags', 'latest_tag'])
        # uid, label
        tmp_label = 1 if sum(data['tag'][index_tag:index_tag+n])>0 else 0
        feature.extend([uid, tmp_label])
        fnames.extend(['uid', 'label'])
        # return
        yield (fnames, feature)

def split_data_long(rs, params, day=21, m=5, n=5):
    #print('loading data')
    #rs = load_data()
    print('getting feature')
    feature_data = []
    feature_train = []
    feature_test = []

    '''
    for uid, data in rs.items():
        if data['userdata'][0] <= day:
            gg = get_feature_long(uid,data,m,n)
            for fname, feature in gg:
                feature_data.append(feature)
            
    df = pd.DataFrame(data=feature_data, columns=fname)
    df_train, df_test = train_test_split(df, test_size=0.2)
    '''
    for uid, data in rs.items():
        if data['userdata'][0] == day:
            gg = get_feature_long(uid,data,m,n)
            for fname, feature in gg:
                feature_test.append(feature)
        elif data['userdata'][0] < day:
            gg = get_feature_long(uid,data,m,n)
            for fname, feature in gg:
                feature_train.append(feature)
    df_train = pd.DataFrame(data=feature_train, columns=fname)
    df_test = pd.DataFrame(data=feature_test, columns=fname)

    train_set = lgb.Dataset(df_train[fname[:-2]], label=df_train['label'])
    num_round = 100
    print('training')
    clf = lgb.train(params, train_set, 100)
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
    print('F',len(Fo)/n1)
    #print(Fx)
    #print(uids_10)
    #print(uids_01)

def get_feature_short(uid, data, m, n):
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
    feature.extend([user_source, user_device])
    fnames.extend(['user_source', 'user_device'])
    # general feature
    log_length = len(tmp_tag) - tmp_tag.index(1)
    tag_sum = sum(tmp_tag)
    vid_sum = sum(tmp_video)
    act_sum = sum(tmp_act_uniq)
    acted_sum = sum(map(sum, tmp_be_act))
    act_avg1 = act_sum/log_length
    act_avg2 = act_sum/tag_sum
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
    act_all = tmp_act_uniq[index_start:index_tag] 
    feature.extend(act_all)
    fnames.extend(['act_day_{}'.format(i) for i in range(1,m+1)])
    video_all = tmp_video[index_start:index_tag] 
    feature.extend(video_all)
    fnames.extend(['video_day_{}'.format(i) for i in range(1,m+1)])

    tag_gaps = [len(i) for i in ''.join(map(str,tmp_tag[index_start:index_tag])).split('1')]
    tag_tags = [len(i) for i in ''.join(map(str,tmp_tag[index_start:index_tag])).split('0')]
    max_gaps = max(tag_gaps)
    latest_gap = tag_gaps[-1]
    max_tags = max(tag_tags)
    latest_tag = tag_tags[-1]
    feature.extend([max_gaps, latest_gap, max_tags, latest_tag])
    fnames.extend(['max_gaps', 'latest_gap', 'max_tags', 'latest_tag'])
    # uid, label
    tmp_label = 1 if sum(data['tag'][index_tag:index_tag+n])>0 else 0
    #tmp_label = sum(data['tag'][index_tag:index_tag+n])
    feature.extend([uid, tmp_label])
    fnames.extend(['uid', 'label'])
    # return
    return (fnames, feature)

def split_data_short(rs, params, day=21, m=5, n=5):
    #print('loading data')
    #rs = load_data()
    print('getting feature')
    feature_data = []
    feature_train = []
    feature_test = []
    '''
    for uid, data in rs.items():
        if data['userdata'][0] <= day:
            fname, feature = get_feature_short(uid,data,m,n)
            feature_data.append(feature)
            
    df = pd.DataFrame(data=feature_data, columns=fname)
    df_train, df_test = train_test_split(df, test_size=0.1)
    '''
    for uid, data in rs.items():
        if data['userdata'][0] == day:
            fname, feature = get_feature_short(uid,data,m,n)
            feature_test.append(feature)
        elif data['userdata'][0] < day:
            fname, feature = get_feature_short(uid,data,m,n)
            feature_train.append(feature)
    df_train = pd.DataFrame(data=feature_train, columns=fname)
    df_test = pd.DataFrame(data=feature_test, columns=fname)

    train_set = lgb.Dataset(df_train[fname[:-2]], label=df_train['label'])
    num_round = 100
    print('training')
    clf = lgb.train(params, train_set, 100)
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
    print('F',len(Fo)/n1)
    #print(Fx)
    #print(uids_10)
    #print(uids_01)
    
def get_feature_1(uid, data, m, n):
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
    feature.extend([user_source, user_device])
    fnames.extend(['user_source', 'user_device'])
    # general feature
    log_length = len(tmp_tag) - tmp_tag.index(1)
    tag_sum = sum(tmp_tag)
    vid_sum = sum(tmp_video)
    act_sum = sum(tmp_act_uniq)
    acted_sum = sum(map(sum, tmp_be_act))
    act_avg1 = act_sum/log_length
    act_avg2 = act_sum/tag_sum
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
    act_latest = tmp_act_uniq[-1] 
    video_latest = tmp_video[-1]
    feature.extend([act_latest, video_latest])
    fnames.extend(['act_latest', 'video_latest'])
    if m>1:
        act_latest2 = tmp_act_uniq[-2] 
        feature.append(act_latest2)
        fnames.append('act_latest2')
    '''
    act_all = tmp_act_uniq[index_start:index_tag] 
    feature.extend(act_all)
    fnames.extend(['act_day_{}'.format(i) for i in range(1,m+1)])
    '''
    tag_gaps = [len(i) for i in ''.join(map(str,tmp_tag[index_start:index_tag])).split('1')]
    tag_tags = [len(i) for i in ''.join(map(str,tmp_tag[index_start:index_tag])).split('0')]
    max_gaps = max(tag_gaps)
    latest_gap = tag_gaps[-1]
    max_tags = max(tag_tags)
    latest_tag = tag_tags[-1]
    feature.extend([max_gaps, latest_gap, max_tags, latest_tag])
    fnames.extend(['max_gaps', 'latest_gap', 'max_tags', 'latest_tag'])
    # uid, label
    tmp_label = 1 if sum(data['tag'][index_tag:index_tag+n])>0 else 0
    feature.extend([uid, tmp_label])
    fnames.extend(['uid', 'label'])
    # return
    return (fnames, feature)

def split_data(rs, params, day, m, n):
    #print('loading data')
    #rs = load_data()
    print('getting feature')
    feature_data = []
    feature_test = []
    feature_train = []
    '''
    for uid, data in rs.items():
        if data['userdata'][0] <= day:
            fname, feature = get_feature_1(uid,data,m,n)
            feature_data.append(feature)
            
    df = pd.DataFrame(data=feature_data, columns=fname)
    df_train, df_test = train_test_split(df, test_size=0.1)
    '''
    for uid, data in rs.items():
        if data['userdata'][0] == day:
            fname, feature = get_feature_1(uid,data,m,n)
            feature_test.append(feature)
        elif data['userdata'][0] < day:
            fname, feature = get_feature_1(uid,data,m,n)
            feature_train.append(feature)
    df_train = pd.DataFrame(data=feature_train, columns=fname)
    df_test = pd.DataFrame(data=feature_test, columns=fname)

    train_set = lgb.Dataset(df_train[fname[:-2]], label=df_train['label'])
    num_round = 100
    print('training')
    clf = lgb.train(params, train_set, 100)
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
    print('F',len(Fo)/n1)
    #print(Fx)
    #print(uids_10)
    #print(uids_01)
    print(tmp_rs[:10])
    print(tmp_rs[-10:])
    
def short_model(m):
    print('short model ================', m)
    rs = load_data()
    params = dict(
        num_leaves=30,
        num_trees=250,
        objective='binary'
    )
    n = 7
    day = 31-n-m
    #split_data_long2(rs, params, day, m, n)
    print('getting feature')
    feature_data = []
    for uid, data in rs.items():
        if data['userdata'][0] <= day:
            gg = get_feature_long2(uid,data,m,n)
            #gg = get_feature_long(uid,data,m,n)
            for fname, feature in gg:
                feature_data.append(feature)
    df_train = pd.DataFrame(data=feature_data, columns=fname)

    feature_pre = []
    for uid, data in rs.items():
        if data['userdata'][0] <= 31-m:
            gg = get_feature_long2(uid,data,m,n,pre=True)
            #gg = get_feature_long(uid,data,m,n)
            for fname, feature in gg:
                feature_pre.append(feature)
    df_pre = pd.DataFrame(data=feature_pre, columns=fname)
    #df_train, df_test = train_test_split(df, test_size=0.2)
    train_set = lgb.Dataset(df_train[fname[:-2]], label=df_train['label'])
    num_round = 100
    print('training')
    clf = lgb.train(params, train_set, 100)

    prod = list(clf.predict(df_pre[fname[:-1]]))
    tmp_rs = [(df_pre['uid'].iloc[i],df_pre['reg_day'].iloc[i],prod[i]) for i in range(0,len(prod))]
    tmp_rs.sort(key=lambda x:x[2], reverse=True)
    print('saving')
    pk.dump(tmp_rs, open('./data/short_md_rs_{}.pkl'.format(m),'wb'))

def long_model():
    print('long model =====================')
    rs = load_data()
    params = dict(
        num_leaves=30,
        num_trees=250,
        objective='binary'
    )
    m = 10
    n = 7
    day = 31-n-m
    #split_data_long2(rs, params, day, m, n)
    print('getting feature')
    feature_data = []
    for uid, data in rs.items():
        if data['userdata'][0] <= day:
            gg = get_feature_long2(uid,data,m,n)
            #gg = get_feature_long(uid,data,m,n)
            for fname, feature in gg:
                feature_data.append(feature)
    df_train = pd.DataFrame(data=feature_data, columns=fname)

    feature_pre = []
    for uid, data in rs.items():
        if data['userdata'][0] <= 31-m:
            gg = get_feature_long2(uid,data,m,n,pre=True)
            #gg = get_feature_long(uid,data,m,n)
            for fname, feature in gg:
                feature_pre.append(feature)
    df_pre = pd.DataFrame(data=feature_pre, columns=fname)
    #df_train, df_test = train_test_split(df, test_size=0.2)
    train_set = lgb.Dataset(df_train[fname[:-2]], label=df_train['label'])
    num_round = 300
    print('training')
    clf = lgb.train(params, train_set, 100)

    prod = list(clf.predict(df_pre[fname[:-1]]))
    tmp_rs = [(df_pre['uid'].iloc[i],df_pre['reg_day'].iloc[i],prod[i]) for i in range(0,len(prod))]
    tmp_rs.sort(key=lambda x:x[2], reverse=True)
    print('saving')
    pk.dump(tmp_rs, open('./data/long_md_rs.pkl','wb'))

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
    m = 20
    n = 7
    day = 31-n-m
    '''
    print('================================== normal model',m)
    split_data(rs, params, day, m, n)
    print('================================== short model',m)
    split_data_short(rs, params, day, m, n)

    '''
    params = dict(
        num_leaves=30,
        num_trees=250,
        objective='binary'
    )
    print('================================== long2 model',m)
    split_data_long2(rs, params, day, m, n)
    print('================================== long2 model',m)
    split_data_long2(rs, params, day, m, n)
    '''
    print('================================== long model',m)
    split_data_long(rs, params, day, m, n)
    '''
         

if __name__ == '__main__':
    #split_data()
    #long_model()
    #short_model(1)
    test()
    #delete_device_1()

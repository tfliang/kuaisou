import json
import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import lightgbm as lgb


class KsFeature:

    def __init__(self, data_path='./data2'):
        self.f_reg = os.path.join(data_path, 'user_register_log.txt')
        self.f_lau = os.path.join(data_path, 'app_launch_log.txt')
        self.f_vid = os.path.join(data_path, 'video_create_log.txt')
        self.f_act = os.path.join(data_path, 'user_activity_log.txt')
        self.data_path = data_path
        self.uf = {}
        self.df_test = 0
        self.df_pre = 0
        self.wrong = 0
        self.clf = 0
        self.dataA = {}
        self.all_score = []
        self.all_pre = {}
        self.all_clf = []

    def data_A(self):
        data_path = './data'
        f_load = os.path.join(self.data_path, 'all_data.pkl')
        all_data_A = joblib.load(f_load)
        uids = list(all_data_A.keys())
        for k in uids:
            all_data_A[-k] = all_data_A.pop(k)
        self.dataA = all_data_A.copy()
        return all_data_A

    def pre_data(self):
        f_save = os.path.join(self.data_path, 'all_data.pkl')
        all_data = {}
        with open(self.f_reg) as f:
            for myline in f:
                myl = myline.split('\t')
                uid = int(myl[0])
                all_data.setdefault(uid, {})
                all_data[uid]['reg_day'] = int(myl[1])
                all_data[uid]['source'] = int(myl[2])
                all_data[uid]['device'] = int(myl[3])
        with open(self.f_lau) as f:
            for myline in f:
                myl = myline.split('\t')
                uid = int(myl[0])
                day = int(myl[1])
                all_data[uid].setdefault('lau', np.zeros(30, dtype=int))
                all_data[uid]['lau'][day-1] = 1
        with open(self.f_vid) as f:
            for myline in f:
                myl = myline.split('\t')
                uid = int(myl[0])
                day = int(myl[1])
                all_data[uid].setdefault('vid', np.zeros(30, dtype=int))
                all_data[uid]['vid'][day-1] += 1
        uids = set(all_data.keys())
        with open(self.f_act) as f:
            for myline in f:
                myl = myline.split('\t')
                uid = int(myl[0])
                day = int(myl[1])
                page = int(myl[2])
                vid = int(myl[3])
                aid = int(myl[4])
                act = int(myl[5])
                all_data[uid].setdefault('acts', np.zeros(30, dtype=int))
                all_data[uid]['acts'][day-1] += 1
                all_data[uid].setdefault('page_type', np.zeros([5,30], dtype=int))
                all_data[uid]['page_type'][page, day-1] += 1
                all_data[uid].setdefault('act_type', np.zeros([6,30], dtype=int))
                all_data[uid]['act_type'][act, day-1] += 1
                if aid in uids:
                    all_data[aid].setdefault('be_act', np.zeros([6,30], dtype=int))
                    all_data[aid]['be_act'][act, day-1] += 1
        for uid, data in all_data.items():
            labels = (
                data.get('acts', np.zeros(30, dtype=int)) + 
                data.get('vid', np.zeros(30, dtype=int)) + 
                data['lau']
            )
            all_data[uid]['label'] = [1 if tmp>0 else 0 for tmp in labels]
        joblib.dump(all_data, f_save)
    
    def load_data(self):
        f_load = os.path.join(self.data_path, 'all_data.pkl')
        all_data = joblib.load(f_load)
        return all_data

    def user_feature(self, all_data):
    # 注意防止信息穿越
        uf = {}
        uf['name'] = ['source', 'device', 'reg_day']
        for uid, data in all_data.items():
            source = data['source']
            device = data['device']
            reg_day = data['reg_day']
            uf[uid] = [source, device, reg_day]
        uf['name'].append('avg_lau')
        for uid, data in all_data.items():
            index_start = list(data['lau']).index(1)
            avg_lau = sum(data['lau'][index_start:]) / (len(data['lau'][index_start:]))
            uf[uid].append(avg_lau)
        uf['name'].extend(['avg_acts', 'avg_acts2'])
        for uid, data in all_data.items():
            index_start = list(data['lau']).index(1)
            avg_acts = sum(data.get('acts', np.zeros(30, dtype=int))[index_start:]) / (sum(data['label'][index_start:]))
            avg_acts2 = sum(data.get('acts', np.zeros(30, dtype=int))[index_start:]) / (len(data['lau'][index_start:]))
            uf[uid].extend([avg_acts, avg_acts2])
        uf['name'].append('vid_act')
        for uid, data in all_data.items():
            vids = sum(data.get('vid',[0]))
            acts = sum(data.get('acts',[0]))
            if vids == 0:
                baa = 0
            elif acts == 0:
                baa = 0
            else:
                baa = vids / acts
            uf[uid].append(baa)
        uf['name'].extend(['page_type_percent_{}'.format(i) for i in range(0,5)])
        for uid, data in all_data.items():
            acts = sum(data.get('acts',[0]))
            if acts > 0:
                for i in range(0,5):
                    acts_sub = sum(data['page_type'][i,:])
                    uf[uid].append(acts_sub / acts)
            else:
                uf[uid].extend([0 for i in range(0,5)])
        uf['name'].extend(['act_type_percent_{}'.format(i) for i in range(0,6)])
        for uid, data in all_data.items():
            acts = sum(data.get('acts',[0]))
            if acts > 0:
                for i in range(0,6):
                    acts_sub = sum(data['act_type'][i,:])
                    uf[uid].append(acts_sub / acts)
            else:
                uf[uid].extend([0 for i in range(0,6)])
        uf['name'] = ['u_'+tmp for tmp in uf['name']]
        if len(uf['name']) != len(uf[uid]):
            print(uf['name'])
            print(uf[uid])
            raise Exception("name != feature")
        self.uf = uf
        return uf

    def log_feature(self, data, index_start, index_end):
    # 需要注意data的长度对特征的影响
        lf = {'name':[], 'feature':[]}
        lf['name'].append('length')
        log_len = index_end - index_start
        lf['feature'].append(log_len)
        lf['name'].extend(['avg_lau', 'avg_acts', 'avg_acts2'])
        acts_part = data.get('acts', np.zeros(30, dtype=int))[index_start:index_end]
        vid_part = data.get('vid', np.zeros(30, dtype=int))[index_start:index_end]
        avg_lau = sum(data['lau'][index_start:index_end]) / log_len
        avg_acts = sum(acts_part) / sum(data['label'][index_start:index_end])
        avg_acts2 = sum(acts_part) / log_len
        lf['feature'].extend([avg_lau, avg_acts, avg_acts2])
        lf['name'].append('sum_vid')
        lf['feature'].append(sum(vid_part))
        lf['name'].append('acts_tail_long')
        weight = np.array([i for i in range(0, index_end - index_start)])
        weight = (weight + 1)**3
        weight = weight / sum(weight)
        long_short = 1
        if log_len >= long_short:
            lf['feature'].append(np.dot(weight, acts_part))
        else:
            lf['feature'].append(np.nan)
        lf['name'].append('acts_tail_short')
        if log_len >= long_short:
            lf['feature'].append(np.nan)
        else:
            lf['feature'].append(np.dot(weight, acts_part))
        lf['name'].append('acts_tail2')
        lf['feature'].append(
            np.dot(weight, np.where(acts_part > 5, 1, 0)))
        lf['name'].append('label_tail')
        lf['feature'].append(np.dot(weight, data['label'][index_start:index_end]))
        lf['name'].extend(['gap_tail', 'gap_max', 'ngp_tail', 'ngp_max'])
        labels = list(data['label'][index_start:index_end])
        labels.reverse()
        if 1 in labels:
            gap_tail = labels.index(1) + 1
        else:
            gap_tail = 0
        if 0 in labels:
            ngp_tail = labels.index(0) + 1
        else:
            ngp_tail = 0
        gap_max = max(map(len, ''.join(map(str, labels)).split('1')))
        ngp_max = max(map(len, ''.join(map(str, labels)).split('0')))
        lf['feature'].extend([gap_tail, gap_max, ngp_tail, ngp_max])
        lf['name'] = ['log_'+tmp for tmp in lf['name']]
        if len(lf['name']) != len(lf['feature']):
            print(lf)
            raise Exception("name != feature")
        return lf

    def day_feature(self):
        pass
    
    def train_set(self, all_data, min_log, num_pre, repeat=True):
        if len(self.uf) >= len(all_data):
            uf = self.uf
        else:
            uf = self.user_feature(all_data)
        train_set = []
        for uid, data in all_data.items():
            label = data['label']
            index_start = label.index(1)
            all_log = len(label) - index_start
            if repeat:
                if all_log >= (min_log + num_pre):
                    min_end = index_start + min_log
                    max_end = len(label) - num_pre
                    for index_end in range(min_end, max_end+1):
                        lf = self.log_feature(data, index_start, index_end)
                        train_label = [1 if sum(label[index_end : index_end + num_pre]) > 0 else 0]
                        train_set.append(
                            uf[uid] + lf['feature'] + [uid] + train_label
                        )
            else:
                if all_log >= (min_log + num_pre):
                    index_end = index_start + min_log
                    lf = self.log_feature(data, index_start, index_end)
                    train_label = [1 if sum(label[index_end : index_end + num_pre]) > 0 else 0]
                    train_set.append(
                        uf[uid] + lf['feature'] + [uid] + train_label
                    )
        name = uf['name'] + lf['name'] + ['uid', 'train_label']
        df_train = pd.DataFrame(data=train_set, columns=name)
        return df_train

    def pre_set(self, all_data, min_log):
        if len(self.uf) >= len(all_data):
            uf = self.uf
        else:
            uf = self.user_feature(all_data)
        pre_set = []
        for uid, data in all_data.items():
            label = data['label']
            index_start = label.index(1)
            all_log = len(label) - index_start
            if all_log >= min_log:
                index_end = len(label)
                lf = self.log_feature(data, index_start, index_end)
                pre_set.append(
                    uf[uid] + lf['feature'] + [uid]
                )
        name = uf['name'] + lf['name'] + ['uid']
        df_pre = pd.DataFrame(data=pre_set, columns=name)
        return df_pre

    def submit_uid(self, dataB, dataA, min_log, num_pre, num_tree=100, fname='uids_b_r2.csv', ntop=40000, repeat=True):
        dataB = self.filter(dataB)
        dataA = self.filter(dataA)
        print('training...')
        clf, score = self.train_model(dataB, dataA, min_log, num_pre, num_tree, split=False, repeat=repeat)
        self.clf = clf
        print('predicting...')
        df = self.pre_set(dataB, min_log=1)
        self.df_pre = df
        feature_name = list(clf.feature_name())
        pre_df = clf.predict(df[feature_name])
        pre = [(int(df.iloc[i]['uid']), pre_df[i]) for i in range(0, len(pre_df))]
        pre.sort(key=lambda x:x[1], reverse=True)
        with open(fname, 'w') as f:
            for tmp in pre[:ntop]:
                f.write(str(tmp[0])+'\n')
        print('finished')
        #return clf

    def filter(self, all_data):
        new_data = all_data.copy()
        ks = list(all_data.keys())
        for k in ks:
            if all_data[k]['device'] == 1:
                tmp = new_data.pop(k)
        return new_data

    def train_model(self, dataB, dataA, min_log, num_pre, num_tree=100, split=True, repeat=True):
        all_data = {}
        all_data.update(dataB)
        all_data.update(dataA)
        df = self.train_set(all_data, min_log, num_pre, repeat=repeat)
        df_train, df_test = train_test_split(df, test_size=0.1)
        if not split:
            df_train = df
        use_feature = [
            'u_source',
            #'u_device',
            #'u_reg_day',
            #'u_avg_lau',
            #'u_avg_acts',
            #'u_avg_acts2',
            #'u_vid_act',
        ]
        #use_feature.extend(['u_page_type_percent_{}'.format(i) for i in range(0,5)])
        #use_feature.extend(['u_act_type_percent_{}'.format(i) for i in range(0,6)])
        use_feature.extend([
            'log_length',
            'log_avg_lau',
            'log_avg_acts',
            'log_avg_acts2',
            'log_sum_vid',
            'log_acts_tail_long',
            'log_acts_tail_short',
            'log_acts_tail2',
            'log_label_tail',
            'log_gap_tail',
            'log_gap_max',
            'log_ngp_tail',
            'log_ngp_max',
        ])
        category_name = [
            'u_source',
            #'u_device',
            #'u_reg_day',
        ]
        ds_train = lgb.Dataset(
            df_train[use_feature], 
            label=df_train['train_label'],
            categorical_feature=category_name,
        )
        ds_test = lgb.Dataset(
            df_test[use_feature],
            label=df_test['train_label'],
            categorical_feature=category_name,
            reference=ds_train,
        )
        params = dict(
            num_leaves=14,
            max_depth=4,
            min_data_in_leaf=200,
            num_trees=num_tree,
            early_stopping_rounds=5,
            bagging_fraction=0.8,
            bagging_freq=10,
            bagging_seed=201,
            lambda_l2=0.02,
            #num_boost_round=100,
            objective='binary',
            seed=2018,
            learning_rate=0.08,
            num_threads=0,
            #verbosity=0,
        )
        clf = lgb.train(params, ds_train, valid_sets=[ds_test])
        score = self.model_score(clf, df_test, use_feature)
        return (clf, score)

    def model_score(self, clf, df_test, name):
        df_pre = clf.predict(df_test[name])
        self.df_test = df_test
        pre = [(
            df_pre[i],
            int(df_test.iloc[i]['train_label']),
            int(df_test.iloc[i]['uid']))
            for i in range(0,len(df_pre))]
        pre.sort(key=lambda x: x[0], reverse=True)
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
        self.wrong = wrong_pre2
        #print(wrong_pre)
        F_score = u1_pre/u1
        F_score2 = 2*u11/(u1 + u11 + u01)
        print(F_score, F_score2)
        print(u11, u10, u00, u01, len(pre))
        return (F_score, F_score2, u11, u11+u01)

    def load_user_type(self):
        f_load = os.path.join(self.data_path, 'user_type.pkl')
        user_type = joblib.load(f_load)
        return user_type
    
def test():
    kk = KsFeature()
    kk.pre_data()
    datab = kk.load_data()
    rs = kk.train_model(datab, {}, min_log=10, num_pre=7, repeat=False)    

if __name__ == '__main__':
    test()

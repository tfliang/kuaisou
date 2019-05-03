import json
import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib


Data_path = './data'
Data_path = './data2'
f_reg = os.path.join(Data_path, 'user_register_log.txt')
f_lau = os.path.join(Data_path, 'app_launch_log.txt')
f_vid = os.path.join(Data_path, 'video_create_log.txt')
f_act = os.path.join(Data_path, 'user_activity_log.txt')
f_save = os.path.join(Data_path, 'pre_data_2.json')


def get_data(rs={}):
    with open(f_reg) as f:
        for myline in f:
            myl = myline.split('\t')
            uid = int(myl[0])
            day = int(myl[1])
            source = int(myl[2])
            device = int(myl[3])
            rs.setdefault(uid,{})['userdata'] = (day,source,device)
            rs.setdefault(uid,{})['tag'] = [0 for i in range(0,30)]
            rs.setdefault(uid,{})['video'] = [0 for i in range(0,30)]
            rs.setdefault(uid,{})['act_page'] = [[0,0,0,0,0] for i in range(0,30)]
            rs.setdefault(uid,{})['act_uniq'] = [set() for i in range(0,30)]
            rs.setdefault(uid,{})['act_act'] = [[0,0,0,0,0,0] for i in range(0,30)]
            rs.setdefault(uid,{})['be_act'] = [[0,0,0,0,0,0] for i in range(0,30)]
            rs[uid]['tag'][day-1] = 1
    with open(f_lau) as f:
        for myline in f:
            myl = myline.split('\t')
            uid = int(myl[0])
            day = int(myl[1])
            rs[uid]['tag'][day-1] = 1
    with open(f_vid) as f:
        for myline in f:
            myl = myline.split('\t')
            uid = int(myl[0])
            day = int(myl[1])
            rs[uid]['video'][day-1] += 1
    with open(f_act) as f:
        for myline in f:
            myl = myline.split('\t')
            uid = int(myl[0])
            day = int(myl[1])
            page = int(myl[2])
            vid = int(myl[3])
            aid = int(myl[4])
            act = int(myl[5])
            rs[uid]['act_page'][day-1][page] += 1
            rs[uid]['act_act'][day-1][act] += 1
            rs[uid]['act_uniq'][day-1].add(vid)
            if rs.get(aid):
                rs[aid]['be_act'][day-1][act] += 1
    for uid in rs.keys():
        rs[uid]['act_uniq'] = [len(tmp) for tmp in rs[uid]['act_uniq']]
    with open(f_save,'w') as f:
        for k,v in rs.items():
            f.write(json.dumps({k:v})+'\n')
    joblib.dump(rs, os.path.join(data_path, '2_pre_data.pkl'))
    return rs


if __name__ == '__main__':
    rs = {}
    rs = get_data(rs)

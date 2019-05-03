import json
import os
import numpy as np


Data_path = './data'
f_reg = os.path.join(Data_path, 'user_register_log.txt')
f_lau = os.path.join(Data_path, 'app_launch_log.txt')
f_vid = os.path.join(Data_path, 'video_create_log.txt')
f_act = os.path.join(Data_path, 'user_activity_log.txt')
f_save = os.path.join(Data_path, 'pre_data.json')


def get_reg(rs):
    with open(f_reg) as f:
        for myline in f:
            myl = myline.split('\t')
            uid = int(myl[0])
            day = int(myl[1])
            source = int(myl[2])
            device = int(myl[3])
            rs.setdefault(uid,{})['day'] = day
            rs.setdefault(uid,{})['source'] = source
            rs.setdefault(uid,{})['device'] = device
    return rs


def get_lau(rs):
    with open(f_lau) as f:
        for myline in f:
            myl = myline.split('\t')
            uid = int(myl[0])
            day = int(myl[1])
            rs[uid].setdefault('launch',[]).append(day)
    return rs
            

def get_vid(rs):
    with open(f_vid) as f:
        for myline in f:
            myl = myline.split('\t')
            uid = int(myl[0])
            day = int(myl[1])
            rs[uid].setdefault('video',[]).append(day)
    return rs


def get_act(rs):
    with open(f_act) as f:
        for myline in f:
            myl = myline.split('\t')
            uid = int(myl[0])
            day = int(myl[1])
            page = int(myl[2])
            vid = int(myl[3])
            aid = int(myl[4])
            act = int(myl[5])
            tmp = (page,vid,aid,act)
            rs[uid].setdefault('action',{}).setdefault(day,[]).append(tmp)
    return rs


def save_data(rs):
    with open(f_save, 'w') as f:
        for k, v in rs.items():
            tmp = {k:v}
            labels = v.get('launch',[]) + v.get('video',[]) + list(v.get('action',{}).keys())
            labels.append(v['day'])
            labels = list(set(labels))
            labels.sort()
            tmp[k]['labels'] = labels

            if v.get('launch'):
                tmp[k]['launch'].sort()
            if v.get('video'):
                vdata = list(zip(*np.unique(tmp[k]['video'],return_counts=True)))
                vdata.sort(key=lambda x: x[0])
                tmp[k]['video'] = [(int(tmp[0]), int(tmp[1])) for tmp in vdata]

            f.write(json.dumps(tmp)+'\n')


if __name__ == '__main__':
    rs = {}
    rs = get_reg(rs)
    rs = get_lau(rs)
    rs = get_vid(rs)
    rs = get_act(rs)
    save_data(rs)

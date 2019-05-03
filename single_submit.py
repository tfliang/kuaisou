import numpy as np
from sklearn.externals import joblib

#pre = joblib.load('all_pre2.pkl')
#pre = joblib.load('all_pre_limit.pkl')

def sub(pre):
    rs = {}
    num = 0
    for m in range(1, 11):
        rs[m] = [tmp[0] for tmp in pre[m] if tmp[1]>=0.5]
        print(m, len(pre[m]), len(rs[m]), len(rs[m])/len(pre[m]))
        num += len(rs[m])

    for m in range(11, 31-7):
        rs[m] = [tmp[0] for tmp in pre[m] if tmp[1]>=0.5]
        print(m, len(pre[m]), len(rs[m]), len(rs[m])/len(pre[m]))
        num += len(rs[m])

    for m in range(24, 31):
        rs[m] = [tmp[0] for tmp in pre[m] if sum(tmp[-10:])/10 >= 0.5]
        print(m, len(pre[m]), len(rs[m]), len(rs[m])/len(pre[m]))
        num += len(rs[m])
    print(num)
    return rs

def sub_rule(pre, datab):
    rs = {}
    num = 0
    for m in range(1, 11):
        rs[m] = [tmp[0] for tmp in pre[m] if tmp[1]>=0.5]
        print(m, len(rs[m]), len(pre[m]), len(rs[m])/len(pre[m]))
        num += len(rs[m])

    for m in range(11, 31):
        day = 31 - m
        uids = [k for k, v in datab.items()
            if v['reg_day']==day and 
            sum(v.get('lau',np.zeros(30,dtype=int))[-5:])>0
        ]
        rs[m] = uids
        print(m, len(uids))
        num += len(uids)

    '''
    for m in range(11, 31-7):
        rs[m] = [tmp[0] for tmp in pre[m] if tmp[1]>=0.5]
        print(m, len(pre[m]), len(rs[m]), len(rs[m])/len(pre[m]))
        num += len(rs[m])

    for m in range(24, 31):
        rs[m] = [tmp[0] for tmp in pre[m] if sum(tmp[-10:])/10 >= 0.5]
        print(m, len(pre[m]), len(rs[m]), len(rs[m])/len(pre[m]))
        num += len(rs[m])
    '''
    print(num)
    return rs

if __name__ == '__main__':
    rs = sub(pre)

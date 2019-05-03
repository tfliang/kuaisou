import pickle as pk
import json


def get_uids():
    rs = []
    pre_num = pk.load(open('./data/pre_num.pkl','rb'))

    uid0 = pk.load(open('./data/long_md_rs.pkl','rb'))
    for day in range(1, 22):
        rs.extend(get_head_day(uid0, pre_num[day][2], day))
        print(day,len(rs),pre_num[day][2],pre_num[day][0])

    for day in range(22,31):
        mm = 31 - day
        uids = pk.load(open('./data/short_md_rs_{}.pkl'.format(mm),'rb'))
        rs.extend(get_head_day(uids, pre_num[day][2],day))
        print(day,len(rs),pre_num[day][2],pre_num[day][0])
    return

    with open('uids_666.csv','w') as f:
        for tmp in rs:
            f.write(str(tmp)+'\n')

def get_lgb():
    rs = []
    pre_num = pk.load(open('./data/pre_num.pkl','rb'))

    uid0 = pk.load(open('./data/lgb_uids.pkl','rb'))

    m = 3
    for day in range(1, 31-m):
        rs.extend(get_head_day(uid0, pre_num[day][2], day))
        print(day,len(rs),pre_num[day][2],pre_num[day][1],pre_num[day][0])
    #for mm in range(1, 2):
    for mm in range(1, m+1):
        day = 31 - mm
        uid_short = pk.load(open('./data/3short_{}.pkl'.format(mm),'rb'))
        rs.extend(get_head_day(uid_short, pre_num[day][2], day))
        print(day,len(rs),pre_num[day][2],pre_num[day][1],pre_num[day][0])

    with open('uids_999.csv','w') as f:
        for tmp in rs:
            f.write(str(tmp)+'\n')


def get_head_day(uid, ntop, day):
    rs = []
    #print(ntop, day)
    #print(len(uid))
    
    for u, dd, pp in uid:
        if int(dd)==day:
            #print(u,dd,pp)
            rs.append(u)
    '''
    print(day, ntop)
    print(uid[:5])
    '''
            
    return rs[:ntop]

if __name__=='__main__':
    get_lgb()
    #get_uids()

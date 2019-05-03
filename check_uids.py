import pickle as pk


def ck():
    rs = pk.load(open('./data/2_pre_data.pkl','rb'))
    pre_num = pk.load(open('./data/pre_num.pkl','rb'))
    uids = {}
    with open('./submit/uids_5.csv') as f:
        for myline in f:
            myl = int(myline)
            day = rs[str(myl)]['tag'].index(1) + 1
            uids.setdefault(day, 0)
            uids[day] += 1
    for i in range(1,31):
        print(pre_num[i], uids[i])
    


if __name__ == '__main__':
    ck()

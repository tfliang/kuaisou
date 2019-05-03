import json
import plotly.offline as pyoff
import plotly.graph_objs as go
import csv
import numpy as np
import pickle as pk
import time
import os
from sklearn.externals import joblib


data_path = './data2'
data_path = './data'
f_data = os.path.join(data_path, '2_pre_data.json')
f_pk = os.path.join(data_path, '2_pre_data.pkl')

def load_data():
    rs = pk.load(open(f_pk,'rb'))
    '''
    rs = {}
    with open(f_data) as f:
        for myline in f:
            myl = json.loads(myline)
            for k,v in myl.items():
                rs[k] = v
    '''
    return rs
           
# fig day vs count(register user)
def fig_reg(rs):
    days = [tmp['userdata'][0] for tmp in rs.values()]
    x = [i for i in range(1,31)]
    y = [days.count(i) for i in x]
    mydata = [go.Bar(x=x, y=y)]
    mylay = go.Layout(title='day-reg',font=dict(size=25),
                      xaxis=dict(title='day'),
                      yaxis=dict(title='reg_count'),
                      bargap=0.1,
                      margin=dict(l=150),
                      showlegend=False)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig)
    
# fig day vs count(register user) grouped by source
def fig_reg_source(rs):
    days = [(tmp['userdata'][0], tmp['userdata'][1]) for tmp in rs.values()]
    source = set([tmp[1] for tmp in days])

    mydata = []
    x = [i for i in range(1,31)]
    for s in source:
        tmp_days = [tmp[0] for tmp in days if tmp[1]==s]
        y = [tmp_days.count(i) for i in x]
        mydata.append(go.Bar(x=x, y=y, name=s))
    mylay = go.Layout(title='day-reg',font=dict(size=25),
                      xaxis=dict(title='day'),
                      yaxis=dict(title='reg_count'),
                      bargap=0.1,
                      margin=dict(l=150),
                      showlegend=True)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='reg_source.png')

def fig_reg_device(rs):
    days = [(tmp['userdata'][0], tmp['userdata'][2]) for tmp in rs.values()]
    source = set([tmp[1] for tmp in days])
    tmp_v = []
    mydata = []
    x = [i for i in range(1,31)]
    for s in source:
        tmp_days = [tmp[0] for tmp in days if tmp[1]==s]
        y = [tmp_days.count(i) for i in x]
        tmp_v.append((s, sum(y)))
        if sum(y) >1000:
            mydata.append(go.Bar(x=x, y=y, name=s))
    mylay = go.Layout(title='day-reg',font=dict(size=25),
                      xaxis=dict(title='day'),
                      yaxis=dict(title='reg_count'),
                      bargap=0.1,
                      margin=dict(l=150),
                      showlegend=True)
    '''
    tmp_v.sort(key=lambda x: x[0])
    #print(tmp_v)
    x = [tmp[0] for tmp in tmp_v]
    y = [tmp[1] for tmp in tmp_v]
    mydata = [go.Bar(x=x, y=y)]
    tmp_v.sort(key=lambda x: x[1], reverse=True)
    #print(tmp_v)
    '''
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='reg_source.png')

# fig day vs count(active user in day)
def fig_tag(rs):
    tags = [0 for i in range(1,31)]
    tags = np.array(tags)
    for uid, data in rs.items():
        tags += data['tag']
    x = [i for i in range(1,31)]
    y = list(tags)
    mydata = [go.Bar(x=x, y=y)]
    mylay = go.Layout(title='day-lab',font=dict(size=25),
                      xaxis=dict(title='day'),
                      yaxis=dict(title='lab_count'),
                      bargap=0.1,
                      margin=dict(l=150),
                      showlegend=True)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='lab_count.png')
       
def fig_act_page(rs):
    acts = {k:np.array([0,0,0,0,0]) for k in range(1,31)}
    for uid, data in rs.items():
        for k, v in enumerate(data['act_page']):
            acts[k+1] += np.array(v)
    x = [i for i in range(1,31)]
    mydata = []
    for k in range(0,5):
        y = [acts[day][k] for day in x]
        name = 'page={}'.format(k)
        mydata.append(go.Bar(x=x,y=y,name=name))
    mylay = go.Layout(title='day-act-page',font=dict(size=25),
                      xaxis=dict(title='day'),
                      yaxis=dict(title='act_page'),
                      bargap=0.1,
                      margin=dict(l=150),
                      showlegend=True)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='act_page.png')

def fig_act_unique(rs):
    all_acts = np.array([0 for i in range(1,31)])
    uniq_acts = np.array([0 for i in range(1,31)])
    for uid, data in rs.items():
        acts_a = np.array(list(map(sum,data['act_page'])))
        acts_u = np.array(data['act_uniq'])
        all_acts += acts_a
        uniq_acts += acts_u
    
    x = [i for i in range(1,31)]
    mydata = []
    y = list(all_acts)
    mydata.append(go.Bar(x=x,y=y,name='all'))
    y = list(uniq_acts)
    mydata.append(go.Bar(x=x,y=y,name='uniq'))
    mylay = go.Layout(title='day-act-uniq',font=dict(size=25),
                      xaxis=dict(title='day'),
                      yaxis=dict(title='act_uniq'),
                      bargap=0.1,
                      margin=dict(l=150),
                      showlegend=True)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='act_page.png')

def fig_act_act(rs):
    acts = {k:np.array([0,0,0,0,0,0]) for k in range(1,31)}
    for uid, data in rs.items():
        for k, v in enumerate(data['act_act']):
            acts[k+1] += np.array(v)
    x = [i for i in range(1,31)]
    mydata = []
    for k in range(0,5):
        y = [acts[day][k] for day in x]
        name = 'page={}'.format(k)
        mydata.append(go.Bar(x=x,y=y,name=name))
    mylay = go.Layout(title='day-act-act',font=dict(size=25),
                      xaxis=dict(title='day'),
                      yaxis=dict(title='act_page'),
                      bargap=0.1,
                      margin=dict(l=150),
                      showlegend=True)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='act_act.png')

def fig_be_act(rs):
    be_acts = np.zeros(30)
    videos = np.zeros(30)
    for uid, data in rs.items():
        be_act = list(map(sum, data['be_act']))
        be_acts += np.array(be_act)
        videos += np.array(data['video'])
    x = [i for i in range(1,31)]
    mydata = []
    y = list(be_acts)
    mydata.append(go.Bar(x=x,y=y,name='be_acts'))
    y = list(videos)
    mydata.append(go.Bar(x=x,y=y,name='videos'))
    mylay = go.Layout(title='day-be-acts',font=dict(size=25),
                      xaxis=dict(title='day'),
                      yaxis=dict(title='be-acts'),
                      bargap=0.1,
                      margin=dict(l=150),
                      showlegend=True)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='act_page.png')

# fig life of user
def fig_life(rs, uid=1260200):
    data = rs[str(uid)]
    x = [i for i in range(1,31)]
    mydata = []
    y = data['tag']
    mydata.append(go.Bar(x=x,y=y,name='tag'))
    y = data['act_uniq']
    mydata.append(go.Bar(x=x,y=y,name='act'))
    y = data['video']
    mydata.append(go.Bar(x=x,y=y,name='video'))
    y = list(map(sum,data['be_act']))
    mydata.append(go.Bar(x=x,y=y,name='be_act'))
    mylay = go.Layout(title='uid={}'.format(uid),font=dict(size=25),
                      xaxis=dict(title='day'),
                      yaxis=dict(title='data'),
                      bargap=0.1,
                      margin=dict(l=150),
                      showlegend=True)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='user.png')

# fig life of user
def fig_life_group(rs):
    # filter
    rs_filter = {}
    for uid, data in rs.items():
        #if data['userdata'][0]==24 and data['userdata'][1]==3:
        if data['userdata'][0]==24 and data['userdata'][2]==1:
        #if data['userdata'][0]==1:
            rs_filter[uid] = data
    S = len(rs_filter)

    x = [i for i in range(1,31)]
    mydata = []
    tags = np.zeros(30)
    uids = {k:set() for k in range(1,31)}
    acts = np.zeros(30)
    vids = np.zeros(30)
    be_acts = np.zeros(30)
    for uid, data in rs_filter.items():
        tags += np.array(data['tag'])
        for d,t in enumerate(data['tag']):
            if t == 1:
                uids[d+1].add(uid)
        acts += np.array(data['act_uniq'])
        vids += np.array(data['video'])
        be_acts += np.array(list(map(sum,data['be_act'])))
    n=15
    uids_d = [0 for i in range(0,n)]
    for i in range(n+1,31):
        before_uids = set()
        for s in range(1,n+1):
            before_uids = before_uids | uids[i-s]
        uids_d.append(len(uids[i] & before_uids))
        print(i,uids_d[i-1],uids[i]-before_uids)

    mydata.append(go.Bar(x=x,y=tags,name='tag'))
    mydata.append(go.Bar(x=x,y=uids_d,name='same'))
    mydata.append(go.Scatter(x=x,y=acts,name='act',yaxis='y2'))
    mydata.append(go.Bar(x=x,y=vids,name='video'))
    mydata.append(go.Bar(x=x,y=be_acts,name='be_act'))
    mylay = go.Layout(title='all {}'.format(S),font=dict(size=25),
                      xaxis=dict(title='day'),
                      yaxis=dict(title='data'),
                      yaxis2=dict(side='right',overlaying='y',range=[0,max(acts)]),
                      bargap=0.1,
                      margin=dict(l=100,r=100),
                      showlegend=True)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='user.png')

def fig_gap(rs):
    gaps = []
    uids = []
    for uid, data in rs.items():
        tag = data['tag']
        tag = ''.join(map(str,tag)).strip('0').split('1')
        tag = [tmp for tmp in tag if len(tmp)>0]
        #tmp = max(map(len,tag))
        tmp = len(tag)
        gaps.append(tmp)
        if tmp == 9:
            print(tag,uid)
            fig_life(rs,uid)
            time.sleep(2)
            
    #gaps.sort(reverse=True)
    #print(gaps[:100])
            #uids.append(uid)
            #print(uid,tmp)
    #gaps = dict(zip(*np.unique(gaps, return_counts=True)))
    #gaps = sorted(gaps.items(), key=lambda x:x[0], reverse=True)
    #print(gaps)
    #return set(uids)

def fig_user(rs):
    fuids = fig_gap(rs)
    device = []
    for uid, data in rs.items():
        if 1:
        #if uid in fuids:
            device.append(data['userdata'][2])
    device = dict(zip(*np.unique(device, return_counts=True)))
    device = sorted(device.items(), key=lambda x:x[1], reverse=True)
    x = [tmp[0] for tmp in device[:10]]
    y = [tmp[1] for tmp in device[:10]]
    mydata = []
    mydata.append(go.Pie(labels=x,values=y))
    mylay = go.Layout(title='device',font=dict(size=25),
                      xaxis=dict(title='device'),
                      yaxis=dict(title='count'),
                      bargap=0.1,
                      margin=dict(l=150),
                      showlegend=True)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='dev.png')

    
if __name__ == '__main__':
    rs = load_data()
    #fig_reg(rs)
    #fig_reg_source(rs)
    fig_reg_device(rs)
    #fig_tag(rs)
    #fig_act_act(rs)
    #fig_act_unique(rs)
    #fig_be_act(rs)
    #fig_life(rs,488089)
    #fig_life(rs,283981)
    #fig_life_group(rs)
    #fig_gap(rs)
    #fig_user(rs)

import json
import plotly.offline as pyoff
import plotly.graph_objs as go
import csv


f_data = './data/pre_data.json'
f_lit = './data/pre_data_lit.json'


# get little file
def lit_data():
    fs = open(f_lit, 'w')
    with open(f_data) as fr:
        for myline in fr:
            myl = json.loads(myline)
            for uid, data in myl.items():
                if data.get('action'):
                    for d,v in data['action'].items():
                        vls = [tmp[-1] for tmp in v]
                        tmp = [vls.count(i) for i in range(0,6)]
                        myl[uid]['action'][d] = tmp
            fs.write(json.dumps(myl)+'\n')
    fs.close()

# get count(register user in reg_day)
def user_day(reg_day=1):
    rs = {}
    with open(f_lit) as f:
        for myline in f:
            myl = json.loads(myline)
            for k,v in myl.items():
                if v['day'] == reg_day:
                    rs[k] = v
    print(len(rs))

# load f_lit
def load_data():
    rs = {}
    with open(f_lit) as f:
        for myline in f:
            myl = json.loads(myline)
            for k,v in myl.items():
                rs[k] = v
    return rs
            
# fig day vs count(register user)
def fig_reg():
    rs = load_data()
    days = [tmp['day'] for tmp in rs.values()]
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
def fig_reg_source():
    rs = load_data()
    days = [(tmp['day'], tmp['source']) for tmp in rs.values()]
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

# fig day vs count(active user in day)
def fig_labels():
    rs = load_data()
    score = {}
    for uid, data in rs.items():
        labels = data['labels']
        for day in labels:
            score[day] = score.setdefault(day,0) + 1
    x = [i for i in range(1,31)]
    y = [score[i] for i in x]
    mydata = [go.Bar(x=x, y=y)]
    mylay = go.Layout(title='day-lab',font=dict(size=25),
                      xaxis=dict(title='day'),
                      yaxis=dict(title='lab_count'),
                      bargap=0.1,
                      margin=dict(l=150),
                      showlegend=True)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='lab_count.png')

# fig day vs count(active user from day to 30)
def fig_labels_2():
    rs = load_data()
    score = {}
    for uid, data in rs.items():
        labels = data['labels']
        labels = [int(tmp) for tmp in data.get('action',{}).keys()]
        labels = data['launch']
        for day in labels:
            score.setdefault(day,[]).append(uid)
    x = [i for i in range(1,31)]
    y = []
    for day in x:
        days = x[day-1:]
        all_uid = []
        for d in days:
            all_uid += score.get(d,[])
        '''
        if day == 28:
            with open('uids_3.csv', 'w') as f:
                for tmp in set(all_uid):
                    f.write(str(tmp)+'\n')
        '''
        y.append(len(set(all_uid)))
        print(day, y[-1])
    mydata = [go.Bar(x=x, y=y)]
    mylay = go.Layout(title='day-lab',font=dict(size=25),
                      xaxis=dict(title='day'),
                      yaxis=dict(title='lab_count'),
                      bargap=0.1,
                      margin=dict(l=150),
                      showlegend=True)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='lab_count.png')

# fig day vs different kind of active 
def fig_labels_3():
    rs = load_data()
    score_lau = {}
    x = [i for i in range(1,31)]
    mydata = []
    for uid, data in rs.items():
        labels = data['launch']
        for day in labels:
            score_lau.setdefault(day,[]).append(uid)
    y = [len(set(score_lau[i])) for i in x]
    mydata.append(go.Bar(x=x, y=y, name='launch'))
    score_act = {}
    for uid, data in rs.items():
        labels = [int(tmp) for tmp in data.get('action',{}).keys()]
        for day in labels:
            score_act.setdefault(day,[]).append(uid)
    y = [len(set(score_act[i])) for i in x]
    mydata.append(go.Bar(x=x, y=y, name='action'))
    score_vid = {}
    for uid, data in rs.items():
        labels = [tmp[0] for tmp in data.get('video',[])]
        for day in labels:
            score_vid.setdefault(day,[]).append(uid)
    y = [len(set(score_vid[i])) for i in x]
    mydata.append(go.Bar(x=x, y=y, name='video'))

    mylay = go.Layout(title='day-lab',font=dict(size=25),
                      xaxis=dict(title='day'),
                      yaxis=dict(title='lab_count'),
                      bargap=0.1,
                      margin=dict(l=150),
                      showlegend=True)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='lab_count.png')
        
# fig day vs percent
def fig_percent():
    rs = load_data()
    score_lau = {}
    x = [i for i in range(1,31)]
    mydata = []
    for uid, data in rs.items():
        labels = data['launch']
        for day in labels:
            score_lau.setdefault(day,[]).append(uid)
    score_act = {}
    for uid, data in rs.items():
        labels = [int(tmp) for tmp in data.get('action',{}).keys()]
        for day in labels:
            score_act.setdefault(day,[]).append(uid)
    score_vid = {}
    for uid, data in rs.items():
        labels = [tmp[0] for tmp in data.get('video',[])]
        for day in labels:
            score_vid.setdefault(day,[]).append(uid)
    
    y = [len(set(score_lau[i])-set(score_act[i])-set(score_vid[i]))/len(set(score_lau[i])) for i in x]
    mydata.append(go.Bar(x=x, y=y, name='only launch'))
    y = [len(set(score_act[i])-set(score_lau[i]))/len(set(score_act[i])) for i in x]
    mydata.append(go.Bar(x=x, y=y, name='action without launch'))
    y = [len(set(score_vid[i])-set(score_lau[i]))/len(set(score_vid[i])) for i in x]
    mydata.append(go.Bar(x=x, y=y, name='video without launch'))
    y = [len(set(score_vid[i])-set(score_act[i]))/len(set(score_vid[i])) for i in x]
    mydata.append(go.Bar(x=x, y=y, name='video without action'))


    mylay = go.Layout(title='day-lab',font=dict(size=25),
                      xaxis=dict(title='day'),
                      yaxis=dict(title='lab_count'),
                      bargap=0.1,
                      margin=dict(l=150),
                      showlegend=True)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='lab_count.png')
     
# fig life of user
def fig_life():
    rs = load_data()
    all_label = {}
    all_reg = {}
    for uid, data in rs.items():
        reg_day = data['day']
        for day in range(reg_day, 31):
            all_reg.setdefault(day-reg_day, []).append(uid)
        for day in data['labels']:
            all_label.setdefault(day-reg_day,[]).append(uid)
    x = [i for i in range(0,30)]
    y = [len(set(all_label.get(i,[]))) for i in x]
    mydata = [go.Bar(x=x, y=y, name='count')]
    y = [len(set(all_label.get(i,[])))/len(set(all_reg.get(i,[]))) for i in x]
    mydata.append(go.Scatter(x=x, y=y, name='percent', yaxis='y2'))
    y = [len(set(all_reg.get(i,[]))) for i in x]
    mydata.append(go.Bar(x=x, y=y, name='all'))
    mylay = go.Layout(title='day-lab',font=dict(size=25),
                      xaxis=dict(title='day'),
                      yaxis=dict(title='lab_count'),
                      yaxis2=dict(title='percent',side='right',overlaying='y'),
                      bargap=0.1,
                      margin=dict(l=150),
                      showlegend=True)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='lab_count.png')
  
# predict
def fig_predict():
    rs = load_data()
    pred = {}
    for uid, data in rs.items():
        reg_day = data['day']
        lab_day = data['labels']
        for day in range(reg_day+1, 31):
            rdays = day - reg_day
            ldays = day - max([i for i in lab_day if i<day])
            c = 1 if day in lab_day else 0
            pred.setdefault('_'.join(map(str,[rdays,ldays])), []).append(c)
    '''
            pred.append((rdays,ldays,c))
    mydata = []
    x = [r[0] for r in pred if r[2]==0]
    y = [r[1] for r in pred if r[2]==0]
    mydata.append(go.Scattergl(x=x,y=y,mode='markers',name='no active',
        marker=dict(opacity=1)))
    x = [r[0] for r in pred if r[2]==1]
    y = [r[1] for r in pred if r[2]==1]
    mydata.append(go.Scattergl(x=x,y=y,mode='markers',name='active',
        marker=dict(opacity=1)))
    '''
    x = []
    y = []
    s = []
    per = []
    for k, v in pred.items():
        rdays, ldays = map(int, k.split('_'))
        x.append(rdays)
        y.append(ldays)
        p = sum(v)/len(v)
        #s.append(p+1)
        s.append(6+p*10)
        per.append(p)
    mydata = []
    mydata.append(go.Scattergl(
        x=x,y=y,mode='markers',
        marker=dict(size=s,color=per,showscale=True),
        hoverinfo='all',
        text=per
        ))

    mylay = go.Layout(title='predict',font=dict(size=25),
                      xaxis=dict(title='have register days'),
                      yaxis=dict(title='last active days'),
                      margin=dict(l=150),
                      showlegend=False)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='lab_count.png')

def user_filter(rs):
    frs = {}
    for uid, data in rs.items():
        if data['source'] not in {0,1,2,3}:
        #if data['source']==3 and data['day']!=24:
            frs[uid] = data
    return frs
      
# predict select user
def fig_predict_filter():
    rs = load_data()
    rs = user_filter(rs)
    pred = {}
    for uid, data in rs.items():
        reg_day = data['day']
        lab_day = data['labels']
        for day in range(reg_day+1, 31):
            rdays = day - reg_day
            ldays = day - max([i for i in lab_day if i<day])
            c = 1 if day in lab_day else 0
            pred.setdefault('_'.join(map(str,[rdays,ldays])), []).append(c)

    x = []
    y = []
    s = []
    per = []
    for k, v in pred.items():
        rdays, ldays = map(int, k.split('_'))
        x.append(rdays)
        y.append(ldays)
        p = sum(v)/len(v)
        #s.append(p+1)
        s.append(6+p*10)
        per.append('_'.join(map(str,[p,sum(v),len(v)])))
    mydata = []
    mydata.append(go.Scattergl(
        x=x,y=y,mode='markers',
        marker=dict(size=s,color=per,showscale=True),
        hoverinfo='all',
        text=per
        ))

    mylay = go.Layout(title='predict',font=dict(size=25),
                      xaxis=dict(title='have register days'),
                      yaxis=dict(title='last active days'),
                      margin=dict(l=150),
                      showlegend=False)
    myfig = go.Figure(data=mydata,layout=mylay)
    pyoff.plot(myfig,image_height=600,image_width=1000,image_filename='lab_count.png')

# cut one user data into many
def get_cut_data(uid, data):
    cut_data = []
    user_data = (uid, data['day'], data['source'])
    exist_days = [i for i in range(data['day'],31)]
    label_days = [1 if i in data['labels'] else 0 for i in exist_days]
    act_days = [sum(data.get('action',{}).get(str(i),[0]))
                for i in exist_days]
    tmp_video = {k:v for k,v in data.get('video',[[-1,-1]])}
    vid_days = [tmp_video.get(i,0) for i in exist_days]
    '''
    print(exist_days)
    print(label_days)
    print(act_days)
    print(vid_days)
    '''
    for pre_delta in range(1, 8):
        #print('=============', pre_delta)
        for preday_index in range(pre_delta, len(exist_days)):
            sup_index = preday_index - pre_delta + 1
            sub_days = exist_days[:sup_index]
            sub_label = label_days[:sup_index]
            sub_act = act_days[:sup_index]
            sub_vid = vid_days[:sup_index]

            label = label_days[preday_index]

            f_all_days = sup_index
            tmp = sub_label[:]
            tmp.reverse()
            f_lact = tmp.index(1) + 1
            f_lact_amount = sub_act[-f_lact]
            f_avg_amount = sum(sub_act)/sum(sub_label)
            f_active_times = sum(sub_label)
            f_avg_act = sum(sub_act)/len(sub_days)
            f_vid = sum(sub_vid)
            feature = (f_all_days, f_lact, f_lact_amount, f_avg_amount, f_active_times, f_avg_act, f_vid, pre_delta, label)
            yield user_data + feature
            
            '''
            print('days',sub_days)
            print('lab',sub_label)
            print('act',sub_act)
            print('vid',sub_vid)
            print('tag',label)
            print('feature',feature)
            '''

# get dataframe for training
def get_dataframe():
    rs = load_data()
    with open('./data/dataset.csv','w') as f:
        fw = csv.writer(f)
        fw.writerow(('uid','reg_day','source',
            'exsit_days','last_act_day','last_act_amount',
            'avg_amount_per_act','active_times','avg_amount_per_day','all_vid',
            'pre_type', 'tag'
            ))
        for uid, data in rs.items():
            for cd in get_cut_data(uid, data):
                fw.writerow(cd)
            #break

# get pre data
def get_pre_data(uid, data):
    cut_data = []
    user_data = (uid, data['day'], data['source'])
    exist_days = [i for i in range(data['day'],31)]
    label_days = [1 if i in data['labels'] else 0 for i in exist_days]
    act_days = [sum(data.get('action',{}).get(str(i),[0]))
                for i in exist_days]
    tmp_video = {k:v for k,v in data.get('video',[[-1,-1]])}
    vid_days = [tmp_video.get(i,0) for i in exist_days]
    '''
    print(exist_days)
    print(label_days)
    print(act_days)
    print(vid_days)
    '''
    #for pre_delta in range(1, 8):
    for pre_delta in range(1, 2):
        sup_index = len(exist_days) 
        sub_days = exist_days[:sup_index]
        sub_label = label_days[:sup_index]
        sub_act = act_days[:sup_index]
        sub_vid = vid_days[:sup_index]

        f_all_days = sup_index
        tmp = sub_label[:]
        tmp.reverse()
        f_lact = tmp.index(1) + 1
        f_lact_amount = sub_act[-f_lact]
        f_avg_amount = sum(sub_act)/sum(sub_label)
        f_active_times = sum(sub_label)
        f_avg_act = sum(sub_act)/len(sub_days)
        f_vid = sum(sub_vid)
        feature = (f_all_days, f_lact, f_lact_amount, f_avg_amount, f_active_times, f_avg_act, f_vid, pre_delta)
        yield user_data + feature
          
# get dataframe for predict
def get_dataset_pre():
    rs = load_data()
    with open('./data/dataset_pre.csv','w') as f:
        fw = csv.writer(f)
        fw.writerow(('uid','reg_day','source',
            'exsit_days','last_act_day','last_act_amount',
            'avg_amount_per_act','active_times','avg_amount_per_day','all_vid',
            'pre_type'
            ))
        for uid, data in rs.items():
            for cd in get_pre_data(uid, data):
                fw.writerow(cd)
            #break


if __name__ == '__main__':
    #fig_reg()
    #fig_reg_source()
    #fig_labels()
    #fig_labels_2()
    #fig_labels_3()
    #fig_percent()
    #fig_life()
    #fig_predict()
    #fig_predict_filter()
    #get_dataframe()
    get_dataset_pre()

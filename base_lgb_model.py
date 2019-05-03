 
import time
import warnings   
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb


warnings.filterwarnings('ignore')
input_path = '../input/'
output_path = '../output/'
windows = [15,2,1]

t0 = time.time()

#%%

def launch_sample(df, begin_date, end_date):
    
    df_sample = df[(df.lau_day > start_day) & (df.lau_day <= end_day)]
    df_sample['label'] = 1
    df_sample = df_sample[['user_id', 'label']].drop_duplicates()
    return df_sample

def register_sample(df, begin_date):
    
    df = user_regis[user_regis.register_day <= start_day]
    df['regis_diff'] = start_day - df['register_day'] + 1
    return df

def user_launch(df, start_day, device_info, i):
    
    df = pd.merge(df,device_info,on=['user_id'],how='left')
    return df[df['lau_day'] <= start_day]
    
def user_video(df, start_day, device_info, i):
    
    df = pd.merge(df,device_info,on=['user_id'],how='left')
    return df[df['vid_day'] <= start_day]
   

def user_action(df, start_day, device_info, i):
    
    df = pd.merge(df,device_info,on=['user_id'],how='left')
    return df[df['act_day'] <= start_day]
  

#%% features
    
def launch_fea1(df,start_day):
  
    result = pd.DataFrame()
    for i in [3,2,1]:
        result_tmp = df[df.lau_day > (start_day - i)]
        result_tmp = result_tmp[['user_id']].drop_duplicates()
        result_tmp['do_lau{}'.format(i)] = 1
        if i == 3:
            result = result_tmp
        else:
            result = pd.merge(result, result_tmp, on = ['user_id'], how = 'left') 
    return result


def launch_fea2(df,start_day):

    result = df.groupby(['user_id'],as_index=False)['lau_day'].agg({
            'recent_lau': 'max'})  
    result['lau_diff'] = start_day - result['recent_lau'] + 1
    return result[['user_id','lau_diff']]


def video_fea1(df,start_day):
    
    result = pd.DataFrame()
    for i in [2,1]:
        result_tmp = df[df.vid_day > (start_day - i)]
        result_tmp = result_tmp[['user_id']].drop_duplicates()
        result_tmp['do_video{}'.format(i)] = 1
        if i == 2:
            result = result_tmp
        else:
            result = pd.merge(result, result_tmp, on = ['user_id'], how = 'left')
    return result


def video_fea2(df,start_day):

    result = df.groupby(['user_id'],as_index=False)['vid_day'].agg({
            'recent_vid': 'max'})  
    result['vid_diff'] = start_day - result['recent_vid'] + 1
    return result[['user_id','vid_diff']]


def action_fea1(df,start_day):
    
    result = pd.DataFrame()
    for i in windows:
        df_tmp = df[df.act_day > (start_day - i)]
        result0 = df_tmp.groupby(['user_id'], as_index=False)['act_day'].agg({
                    'user_act_unique_{}'.format(i): 'nunique',
                     'user_act_last_{}'.format(i): 'max'})
    
        result1 = df_tmp.groupby(['user_id','act_day'],as_index=False)['page'].agg({
                'user_day_act_count': 'count'})
        result2 = result1.groupby(['user_id'],as_index=False)['user_day_act_count'].agg({
                'act_day_count_min_{}'.format(i): 'min',
                'act_day_count_std_{}'.format(i): 'std'})    
        result_tmp = pd.merge(result0,result2,on=['user_id'],how='left')
        if i == windows[0]:
            result = result_tmp
        else:
            result = pd.merge(result, result_tmp, on = ['user_id'], how = 'left')
    return result

            

#%% 测试代码


if __name__ == "__main__":
    
    #初赛数据B
    user_regis = pd.read_csv(input_path + '/data_b/' + 'user_register_log.txt',sep="\s+",header=None)
    user_lau  = pd.read_csv(input_path + '/data_b/' + 'app_launch_log.txt',sep='\s+',header=None)
    user_vid  = pd.read_csv(input_path + '/data_b/' + 'video_create_log.txt',sep='\s+',header=None)
    user_act  = pd.read_csv(input_path +'/data_b/' + 'user_activity_log.txt',sep="\s+",header=None)
    
    
    user_act.columns = ['user_id','act_day','page','video_id','author_id','action_type']
    user_regis.columns = ['user_id','register_day','register_type','device_type']
    user_lau.columns = ['user_id','lau_day']
    user_vid.columns = ['user_id','vid_day']
    device_info = user_regis[['user_id','register_day','device_type']]
   

    
    train = pd.DataFrame() 
    
    for i in tqdm(range(2)): 
        start_day = 30 - i * 7
        end_day = start_day + 7
        
        #%% 生成label
        df_sample = launch_sample(user_lau, start_day, end_day)
        df_regis = register_sample(user_regis,start_day)
        df_tmp = pd.merge(df_regis, df_sample, on = 'user_id', how = 'left')
        df_tmp['label'].fillna(0, inplace=True)
        
        #%% features merge
        tmp_launch = user_launch(user_lau, start_day, device_info, i)
        tmp_video = user_video(user_vid, start_day, device_info, i)
        tmp_action = user_action(user_act,start_day, device_info, i)
        
        df_tmp = pd.merge(df_tmp, launch_fea1(tmp_launch,start_day), on = 'user_id', how = 'left')
        df_tmp = pd.merge(df_tmp, video_fea1(tmp_video,start_day), on = 'user_id', how = 'left')
        df_tmp = pd.merge(df_tmp, action_fea1(tmp_action,start_day), on = 'user_id', how = 'left')
        df_tmp = pd.merge(df_tmp, launch_fea2(tmp_launch,start_day), on = 'user_id', how = 'left')
        df_tmp = pd.merge(df_tmp, video_fea2(tmp_video,start_day), on = 'user_id', how = 'left')

         #%% train,test
        if (i == 0):
             test = df_tmp
        if (i != 0) :
             train = pd.concat((train, df_tmp), axis = 0)
        
            
    #%% 模型
  
    
    params = {'num_leaves': 2**7 - 1,
              'boosting': 'gbdt',
              'objective': 'binary',
              'max_depth': 3,
              'min_child_weight': 1.2,
              'learning_rate': 0.05,
              'feature_fraction': 0.75,
              'bagging_freq': 1,
              'metric': 'auc',
              'num_threads': 4
              }
    
    
    featureName = train.drop(['label'], axis=1).columns.tolist() 
    print(train.shape,test.shape)
    train_X = train.drop(['label'], axis=1).values
    train_y = train['label'].values
    test_X = test.drop(['label'], axis=1).values
    test_y = test['label'].values

       
    lgb_train = lgb.Dataset(train_X, train_y)
    lgb_test = lgb.Dataset(test_X, test_y)

    gbm = lgb.train(params,lgb_train,1000)
    pred = gbm.predict(test_X)
    
    print('训练用时{}分'.format((time.time() - t0) / 60))
    
    submission = pd.DataFrame()
    submission['user_id'] = test['user_id']
    submission['probability'] = pred
    submission = submission.sort_values(['probability'], ascending=False).head(25000)
    submission['user_id'].to_csv('sub.csv',index=False)

    
    
 
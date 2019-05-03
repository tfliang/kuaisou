import numpy as np
import pandas as pd
from pandas import DataFrame
#import xgboost

path ="./"
app = pd.read_table(path+'app_launch_log.txt',names=['user_id','day'],encoding='utf-8',sep='\t',)
user_act = pd.read_table(path+'user_activity_log.txt',names=['user_id','day','page','video_id','author_id','action_type'],encoding='utf-8',sep='\t')
user_reg = pd.read_table(path+'user_register_log.txt',names=['user_id','register_day','register_type','device_type'],encoding='utf-8',sep='\t')
vedio = pd.read_table(path+'video_create_log.txt',names=['user_id','day'],encoding='utf-8',sep='\t')

# split train and valid
#train_act,valid_act = user_act[user_act.day < 24],user_act[user_act.day >= 24]
#train_reg,valid_reg = user_reg[user_reg.register_day < 24],user_reg[user_reg.register_day >= 24]
# Your Feature Engineering Work


# 取最后一个星期有action的用户即可，Baseline F1: 0.790+
userPre = user_act[user_act.day>=24]
sub = userPre[['user_id']].drop_duplicates()
sub.to_csv(path+"base.csv",encoding='utf-8',index=None,header=None)

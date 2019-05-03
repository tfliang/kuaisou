import numpy as np
import pandas as pd
from pandas import DataFrame
import pickle as pk


def pkl_data():
    data_reg = pd.read_table('user_register_log.txt',
        names=['uid','day','source','device'])
    pk.dump(data_reg, open('data_reg.pkl', 'wb'))

    data_launch = pd.read_table('app_launch_log.txt',names=['uid','day'])
    pk.dump(data_launch, open('data_launch.pkl', 'wb'))
    
    data_video = pd.read_table('video_create_log.txt',names=['uid','day'])
    pk.dump(data_video, open('data_video.pkl', 'wb'))

    data_action = pd.read_table('user_activity_log.txt',
        names=['uid','day','page','vid','aid','action'])
    pk.dump(data_action, open('data_action.pkl', 'wb'))
   

def load_data():
    data_reg = pk.load(open('data_reg.pkl', 'rb'))
    data_launch = pk.load(open('data_launch.pkl', 'rb'))
    data_video = pk.load(open('data_video.pkl', 'rb'))
    data_action = pk.load(open('data_action.pkl', 'rb'))
    return (data_reg, data_launch, data_video, data_action)

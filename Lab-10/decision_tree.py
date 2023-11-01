import pandas as pd
import numpy as np
df = pd.read_csv('data.csv')
df

def compute_total_entropy(df):
    total_samples = len(df)
    count_yes,count_no = len(df['Decision'] == 'Yes'),len(df['Decision'] == 'No')
    total_entropy = -(count_yes/total_samples)*np.log2(count_yes/total_samples) - (count_no/total_samples)*np.log2(count_no/total_samples) 
    return total_entropy

def calc_entropy(df,label,class_list):
    class_count = len(df)
    entropy = 0
    for c in class_list:
        label_class_count = df[df[label] == c].shape[0] 
        entropy_class = 0
        if label_class_count != 0:
            p_class = label_class_count/class_count
            entropy_class = -p_class*np.log2(p_class)
        entropy += entropy_class
    return entropy


def calc_info_gain(feature_name,df,label,class_list):
    feature_val_list = df[feature_name].unique()
    total_rows = df.shape[0]
    for feature_val in feature_val_list:
        feature_val_data = df[df[feature_name] == feature_val]
        feature_val_count = feature_val_data.shape[0]
        feature_val_entropy = calc_entropy(feature_val_data,label,class_list)
        feature_val_probability = feature_val_count/total_rows
        feature_info += feature_val_probability*feature_val_entropy
    info_gain  = compute_total_entropy(df) - feature_info
    return info_gain


def find_most_informative_feature(df,label,class_list):
    x = df.columns.drop(label)
    max_info_gain = -1
    max_info_feature = None 
    for feature in x:
        feature_info_gain = calc_info_gain(feature,df,label,class_list)
        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature = feature
    return max_info_feature



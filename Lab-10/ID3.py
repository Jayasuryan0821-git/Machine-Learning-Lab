import numpy as np
import pandas as pd 
df = pd.read_csv('data.csv')
df['Wind'] = df['Wind'].map({'Weak':0,'Strong':1})
df['Decision'] = df['Decision'].map({'No':0,'Yes':1})
df = pd.get_dummies(df,columns=['Outlook'])

def convertTemp(val):
    if val >= df['Temp'].mean():
        return 1
    else:
        return 0

def convertHumidity(val):
    if val >= df['Humidity'].mean():
        return 1
    else:
        return 0

df['Temp'] = df['Temp'].apply(convertTemp)
df['Humidity'] = df['Humidity'].apply(convertHumidity)
df['Outlook_Overcast'] = df['Outlook_Overcast'].map({False:0,True:1})
df['Outlook_Rain'] = df['Outlook_Rain'].map({False:0,True:1})
df['Outlook_Sunny'] = df['Outlook_Sunny'].map({False:0,True:1})
x = df.drop('Decision',axis=1).values
y = df['Decision'].values

class TreeNode:
    def __init__(self,entropy=None,feature=None,sample=None,left=None,right=None,value=None):
        self.entropy = entropy 
        self.feature = feature 
        self.sample = sample
        self.left = left 
        self.right = right 
        self.value = value
    
    def is_leaf(self):
        return self.left is None and self.right is None 

def compute_entropy(y):
    entropy = 0
    if len(y) > 0:
        p1 = len(y[y == 1])/len(y)
        if p1 not in (0,1):
            entropy = -p1*np.log2(p1) - (1-p1)*np.log2(1-p1)
    return entropy 

def split_dataset(x,node_indices,feature):
    left_indices,right_indices = [],[]
    for i in node_indices:
        if x[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices,right_indices

def compute_information_gain(x,y,node_indices,feature):
    left_indices,right_indices = split_dataset(x,node_indices,feature)
    x_node,y_node = x[node_indices],y[node_indices]
    x_left,y_left = x[left_indices],y[left_indices]
    x_right,y_right = x[right_indices],y[right_indices]
    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    w_left = len(x_left)/len(x)
    w_right = len(x_right)/len(x)
    weighted_entropy = w_left*left_entropy + w_right*right_entropy
    information_gain = node_entropy - weighted_entropy
    return information_gain

def get_best_split(x,y,node_indices):
    num_features = x.shape[1]
    best_feature = None
    max_info_gain = -np.inf
    for feature in range(num_features):
        info_gain = compute_information_gain(x,y,node_indices,feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
    return best_feature

def build_tree(x,y,node_indices,current_depth,max_depth):
    entropy = compute_entropy(y[node_indices])
    value = np.bincount(y[node_indices],minlength=2)
    node = TreeNode(entropy=entropy,value=value,sample=len(node_indices)) 
    if current_depth == max_depth or entropy == 0 or len(np.unique(y[node_indices])) == 1:
        return node 
    best_feature = get_best_split(x,y,node_indices)
    if best_feature is None:
        return node 
    node.feature = best_feature
    left_indices,right_indices = split_dataset(x,node_indices,best_feature)
    if left_indices:
        node.left = build_tree(x,y,left_indices,current_depth+1,max_depth)
    if right_indices:
        node.right = build_tree(x,y,right_indices,current_depth+1,max_depth)
    return node

def print_tree(node,feature_names=None,depth=0):
    if feature_names is not None and node.feature is not None:
        feature_name = feature_names[node.feature]
    else:
        feature_name = f'Feature {node.feature}'
    if node.is_leaf():
        print(f"{' '*depth}Leaf node, Class Distrubution:{node.value}")
    else:
        print(f"{' '*depth}{feature_name} (entropy = {node.entropy}) (samples = {node.sample})")
    if node.left:
        print(f"{' '*depth}Left")
        print_tree(node.left,feature_names=feature_names,depth=depth+1)
    if node.right:
        print(f"{' '*depth}Right")
        print_tree(node.right,feature_names=feature_names,depth=depth+1)

def predict(node,sample):
    while not node.is_leaf():
        feature_index = node.feature
        if sample[feature_index] == 1:
            node = node.right
        else:
            node = node.left
    return np.argmax(node.value)

root = build_tree(x,y,list(range(len(y))),0,6)
feature_names = df.drop('Decision',axis=1).columns.to_list()
print_tree(root,feature_names=feature_names,depth=0)
sample_input = [1, 0, 1, 0, 1, 0]
y_pred = predict(root,sample_input)
print(f"Predicted class for input {sample_input}: {y_pred}")

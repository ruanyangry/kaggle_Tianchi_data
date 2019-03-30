# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 08:17:12 2019

@author: RY
@2019/3/30
@https://www.kaggle.com/ruanyang001/anamoly-detection-using-undersampling-and-xgboost/edit
@Purpose: 熟悉半导体行业数据处理方式
"""

# 导入必要的库

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from feature_selector import FeatureSelector
warnings.filterwarnings("ignore",category=DeprecationWarning)

datapath=r"C:\Users\RY\Desktop\Semiconductor\\"

# 利用pandas读取csv格式数据

data=pd.read_csv(datapath+"uci-secom.csv")
origin_data=data
y_labels=data["Pass/Fail"]
data=data.drop(columns=["Pass/Fail"])

# 筛除数据集中的第一列时间戳
data=data.drop(columns=["Time"],axis=1)
label_count=y_labels.value_counts()

print("#-----------------------------------------#")
print("输出分类结果")
print(label_count)
print("#-----------------------------------------#")
print("\n")

print("#-----------------------------------------#")
print("数据维度:{}".format(data.shape))
print("特征总数:{}".format(data.shape[1]))
print("标签维度:{}".format(y_labels.shape[0]))
print("分类结果fail:{}".format(label_count.values[0]))
print("分类结果pass:{}".format(label_count.values[1]))
print("#-----------------------------------------#")
print("\n")

print("#-----------------------------------------#")
print("绘制pie chart查看 pass/fail 数据占比")
labels=["Fail","Pass"]
size=label_count.values
colors=['magenta', 'lightgreen']
explode=[0,0.1]

plt.rcParams["figure.figsize"]=(10,10)
plt.pie(size,labels=labels,colors=colors,explode=explode,autopct="%.2f%%",shadow=True)
plt.axis("off")
plt.title("A pie Chart Representing the no. of tests passes or failed",fontsize=20)
plt.legend()
plt.savefig("fail_pass_pie.jpg",dpi=300)
plt.show()
print("#-----------------------------------------#")

print("#-----------------------------------------#")
print("查看数据集column的数据类型")
data_types=data.dtypes
with open("column_dtype.txt","w") as f:
    for index,data_type in enumerate(data_types):
        f.write("特征:{} 数据类型:{}\n".format(index+1,data_type))
print("#-----------------------------------------#")
print("\n")

print("#-----------------------------------------#")
print("查看数值型数据的分布情况")
columne_value_describe=data.describe()
columne_value_describe.to_csv("columne_value_describe.csv",index=True,header=True)
print("#-----------------------------------------#")
print("\n")

print("#-----------------------------------------#")
print("利用FeatureSelector进行数据预处理")
fs=FeatureSelector(data=data,labels=y_labels)
print("# identify_missing")
fs.identify_missing(missing_threshold=0.65)
missing_features=fs.ops["missing"]
missing_stats=fs.missing_stats
fs.plot_missing()
plt.savefig("missing_features.jpg",dpi=300)
plt.show()

print(fs.missing_stats.head())
print("\n")

print("# identify_single_unique")
fs.identify_single_unique()
single_uniques=fs.ops["single_unique"]
with open("single_unique_feature_count.txt","w") as f:
    for index,single_unique in enumerate(single_uniques):
        f.write("count:{}  single_unique_feature:{}\n".format(index+1,single_unique))
    
fs.plot_unique()
plt.savefig("single_unique_feature.jpg",dpi=300)
plt.show()

print(fs.unique_stats.sample(data.shape[1]))
print("\n")

print("# identify_collinear")
fs.identify_collinear(correlation_threshold=0.98)
correlated_features=fs.ops["collinear"]

with open("collinear.txt","w") as f:
    for index,correlated_feature in enumerate(correlated_features):
        f.write("{}   feature name:{}\n".format(index+1,correlated_feature))
fs.plot_collinear()
plt.savefig("collinear.jpg",dpi=300)
plt.show()

print(fs.record_collinear.head())

print("\n")

print("# identify_zero_importance")
print("使用LightGBM库训练GB集成算法，评价特征之间的重要性")
print("1. 为了降低随机性，模型默认会训练10次")
print("2. 模型默认会采用 early stopping 的操作形式，使用15%的数据作为 validation data 去获取 optimal number of estimators")
print("3. 需要使用到的参数")
print("    task: classification or regression , metrics 与这个是相关的")
print("    eval_metric: 用于 early stopping 的指标，auc for classification, L2 for regression")
print("    n_iterations：训练的次数，默认是10次,feature importances会取10次计算结果的平均值")
print("    early_stopping: 默认在训练的时候是使用 early stopping 模式的，early stopping")
print("                    可以理解成一个 regulation，为了防止训练数据的过拟合")
fs.identify_zero_importance(task="classification",eval_metric="auc",
                            n_iterations=10,early_stopping=True)

zero_importance_features=fs.ops["zero_importance"]
with open("zero_importance.txt","w") as f:
    for index,zero_importance_feature in enumerate(zero_importance_features):
        f.write("特征个数：{}  特征名称：{}\n".format(index,zero_importance_feature))
fs.plot_feature_importances(threshold=0.99,plot_n=20)
plt.savefig("feature_importance.jpg",dpi=300)
plt.show()

one_hundred_features=list(fs.feature_importances.loc[:99,"feature"])

print("\n")

print("# identify_low_importance")
fs.identify_low_importance(cumulative_importance=0.99)
low_importance_features=fs.ops["low_importance"]
with open("low_importance.txt","w") as f:
    for index,low_importance_feature in enumerate(low_importance_features):
        f.write("特征个数：{}  特征名称：{}\n".format(index+1,low_importance_feature))
print("#-----------------------------------------#")
print("\n")

print("#-----------------------------------------#")
print("移除上述方法判断出来的不需要特征")
print("输出需要被移除的特征")
feature_remove=fs.check_removal()
for i in feature_remove:
    print("移除特征:{}".format(i))
data_remove_feature=fs.remove(methods="all")
print("原始特征个数:{}".format(data.shape[1]))
print("当前特征个数:{}".format(data_remove_feature.shape[1]))
print("#-----------------------------------------#")
print("\n")

print("#---------------------------------#")
print("剩下特征缺失值使用0来进行填充")
data=data_remove_feature.replace(np.NaN,0)
if data.isnull().any().any():
    print("数据集中存在数据缺失")
    print(data.shape[0]-data.count())
else:
    print("数据集中不存在参数缺失")
print("#---------------------------------#")
print("\n")

print("#---------------------------------#")
print("特征数据集维度:{}".format(data.shape))
print("标签数据集维度:{}".format(y_labels.shape))
if data.shape[0] == y_labels.shape[0]:
    print("特征维度与标签维度匹配")
else:
    print("特征维度与标签维度不匹配，请检查数据集")
print("#---------------------------------#")
print("\n")

print("#---------------------------------#")
print("划分测试集和训练集")

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data,y_labels,test_size=0.2,random_state=0)

print("训练集维度:{}".format(x_train.shape))
print("训练集标签维度:{}".format(y_train.shape))
print("测试集维度:{}".format(x_test.shape))
print("测试集标签维度:{}".format(y_test.shape))
print("#---------------------------------#")
print("\n")

print("#---------------------------------#")
print("归一化数据集")
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
print("#---------------------------------#")
print("\n")

print("#---------------------------------#")
print("下采样")
origin_data=origin_data.drop(columns=["Time"],axis=1)
failed_tests=np.array(origin_data[origin_data["Pass/Fail"]==1].index)
no_failed_tests=len(failed_tests)
normal_indices=origin_data[origin_data["Pass/Fail"]==-1]
no_normal_indices=len(normal_indices)

print("no_failed_tests={}".format(no_failed_tests))
print("no_normal_indices={}".format(no_normal_indices))

random_normal_indices = np.random.choice(no_normal_indices, size = no_failed_tests, replace = True)
random_normal_indices = np.array(random_normal_indices)

undersample = np.concatenate([failed_tests, random_normal_indices])

data["Pass/Fail"]=y_labels
undersample_data=data.iloc[undersample,:]
print("下采样数据集维度:{}".format(undersample_data.shape))
print("#---------------------------------#")
print("\n")

print("#---------------------------------#")
print("拆分下采样数据集")

x = undersample_data.iloc[:,undersample_data.columns != 'Pass/Fail']
y = undersample_data.iloc[:,undersample_data.columns =='Pass/Fail']

print("下采样特征数据集维度:{}".format(x.shape))
print("下采样标签数据集维度:{}".format(y.shape))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print("训练集维度:{}".format(x_train.shape))
print("训练集标签维度:{}".format(y_train.shape))
print("测试集维度:{}".format(x_test.shape))
print("测试集标签维度:{}".format(y_test.shape))
print("#---------------------------------#")
print("\n")

print("#---------------------------------#")
print("归一化下采样数据集")
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
print("#---------------------------------#")
print("\n")

print("#---------------------------------#")
print("XGB 集成方法分类")

from xgboost.sklearn import XGBClassifier

model=XGBClassifier()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("计算AUPRC得分")
print("横轴: recall，纵轴: precision,AUPRC对应的是曲线下的面积")

from sklearn.metrics import average_precision_score

probabilities=model.fit(x_train,y_train).predict_proba(x_test)

print('AUPRC = {}'.format(average_precision_score(y_test, probabilities[:, 1])))

print("计算confusion_matrix")

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

plt.rcParams["figure.figsize"]=(5,5)
sns.set(style="dark",font_scale=1.4)
sns.heatmap(cm,annot=True,annot_kws={"size":15})
print("#---------------------------------#")
print("\n")

print("#---------------------------------#")
print("超参数调整")
print("1. 格点搜索")

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

parameters=[{"max_depth":[1,4,6,8,10]}]

grid_search=GridSearchCV(estimator=model,param_grid=parameters,scoring="accuracy",
                         cv=5,n_jobs=1)

grid_search=grid_search.fit(x_train,y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print("格点搜索的最佳精度:{}".format(best_accuracy))
print("格点搜索的最佳参数")
print(best_parameters)
print("\n")

#print("2. 随机搜索")

#random_search = RandomizedSearchCV(estimator=model, 
#                                   param_distributions=parameters,
#                                   scoring="accuracy",
#                                   n_iter=10, 
#                                   cv=5, 
#                                   iid=False,
#                                   n_jobs=1)
#random_search=random_search.fit(x_train,y_train)
#
#best_accuracy = random_search.best_score_
#best_parameters = random_search.best_params_
#
#print("随机搜索的最佳精度:{}".format(best_accuracy))
#print("随机搜索的最佳参数")
#print(best_parameters)
#print("\n")
#
#print("3. TPE搜索")
#
#print("#---------------------------------#")
#print("\n")

print("#---------------------------------#")
print("使用格点搜索的最佳参数")
weights=(y==0).sum()/(1.0*(y==-1).sum())
model=XGBClassifier(max_depth=5,scale_pos_weights=weights,n_job=1)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("#---------------------------------#")
print("\n")

import xgboost as xgb

print("#---------------------------------#")
print("绘制特征重要性图")
colors=plt.cm.Set1(np.linspace(0,1,9))
xgb.plot_importance(model,height=1,color=colors,
                    grid=False,importance_type="cover",
                    show_values=False)

plt.rcParams['figure.figsize'] = (30, 40)
plt.title('The Feature Importances in a Model', fontsize = 20)
plt.xlabel('The F-Score for each features')
plt.ylabel('Importances')
plt.savefig("features_importances.jpg",dpi=300)
plt.show()

print("#---------------------------------#")
print("\n")

print("#---------------------------------#")
print("输出 learning_curve")
from sklearn.model_selection import learning_curve

weights = (y == 0).sum()/(1.0*(y == -1).sum())
trainSizes, trainScores, crossValScores = learning_curve(XGBClassifier(max_depth = 5, 
                                                                       scale_pos_weights = weights,
                                                                       n_jobs = 1),
x_train, y_train, scoring = 'average_precision')

trainScoresMean = np.mean(trainScores, axis=1)
trainScoresStd = np.std(trainScores, axis=1)
crossValScoresMean = np.mean(crossValScores, axis=1)
crossValScoresStd = np.std(crossValScores, axis=1)

colours = plt.cm.tab10(np.linspace(0, 1, 9))

fig = plt.figure(figsize = (14, 9))
plt.fill_between(trainSizes, trainScoresMean - trainScoresStd, trainScoresMean + trainScoresStd, alpha=0.1, color=colours[0])
plt.fill_between(trainSizes, crossValScoresMean - crossValScoresStd, crossValScoresMean + crossValScoresStd, alpha=0.1, color=colours[1])


plt.plot(trainSizes, trainScores.mean(axis = 1), 'o-', label = 'train', color = colours[0])
plt.plot(trainSizes, crossValScores.mean(axis = 1), 'o-', label = 'cross-val', color = colours[1])

ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, ['train', 'cross-val'], bbox_to_anchor=(0.8, 0.15), \
               loc=2, borderaxespad=0, fontsize = 16);
plt.xlabel('training set size', size = 16); 
plt.ylabel('AUPRC', size = 16)
plt.title('Learning curves indicate slightly underfit model', size = 20)
plt.savefig("learning_curve.jpg",dpi=300)
plt.show()

print("#---------------------------------#")
print("\n")



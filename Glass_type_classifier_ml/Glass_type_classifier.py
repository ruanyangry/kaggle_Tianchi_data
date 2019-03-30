# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 10:26:18 2019

@author: Ruan Yang
@Reference: https://www.kaggle.com/eliekawerk/glass-type-classification-with-machine-learning
@Purpose: 根据元素含量来实现glass type classifier
@Contents
@1. Prepare Problem
@   load libraries
@   load and explore the shape of the dataset
@2. Summarize Data
@   Descriptive statistics
@   Data visualization
@3. Prepare Data
@   Data Cleaning
@   Split-out validation dataset
@   Data transformation
@4. Evaluate Algorithms
@   Dimensionality reduction
@   Compare Algorithms
@5. Improve Accuracy
@   Algorithm Tuning
@6. Diagnose the performance of the best algorithms
@   Diagnose overfitting by plotting the learning and validation curves
@   Further tuning
@7. Finalize Model
@   Create standalone model on entire training dataset
@   Predictions on test dataset
"""

# prepare problem

# First load necessary library

import os
import numpy as np # linear algebra
import pandas as pd # read and wrangle dataframes
import matplotlib.pyplot as plt # visualization
import seaborn as sns # statistical visualization and aesthetics

# sklearn library
from sklearn.base import TransformerMixin # To create new classes for transformations
from sklearn.preprocessing import FunctionTransformer,StandardScaler # preprocessing
from sklearn.decomposition import PCA # dimensionality reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split,KFold
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.model_selection import GridSearchCV,learning_curve,validation_curve

from scipy.stats import boxcox # data transform

# ules

from sklearn.pipeline import Pipeline # streaming pipelines

# To create a box-cox transformation class

from sklearn.base import BaseEstimator,TransformerMixin

from collections import Counter
import warnings

# load models

from sklearn.tree import DecisionTreeClassifier # 决策树分类器
from sklearn.linear_model import LogisticRegression # logistic 分类器
from xgboost import XGBClassifier,plot_importance # 分类器
from sklearn.svm import SVC # 支持向量机分类
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier # 基于树的集成学习算法
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier # 近邻算法 KNN 分类
from sklearn.naive_bayes import GaussianNB # 高斯型朴素贝叶斯分类器

# 记录程序运行时间

from time import time

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

datapath=r"C:\Users\RY\Desktop\Glass_type_classifier_ml\\"

df=pd.read_csv(datapath+"glass.csv")

features=df.columns[:-1].tolist()

# 输出数据集的维度

print("数据集维度{}".format(df.shape))
print("\n")

# 输出前15个数据进行查看

print("输出数据集前15个数据")
print(df.head(15))
print("\n")

# 查看columns的数据类型

print("输出数据类型{}".format(df.dtypes))
print("\n")

# 查看数值型数据的分布情况

print("查看数值型数据的分布情况")
print(df.describe())
print("\n")

# 通过 df.describe 可以看出不同 columns的数据不处于同一个维度
# 上面，需要进行归一化操作，使数据处于同一个维度之中

# 统计每一个columns数据在整个数据集中出现的次数(这个是建立在初始的
# 数据集中含有较多的 0 的前提下)

print("不同Type的分布情况")
print(df["Type"].value_counts())
print("\n")

# 绘图查看数据的分布情况
# univariate plots 单变量绘图

if os.path.exists("univariate_plots"):
    os.chdir("univariate_plots")
else:
    os.mkdir("univariate_plots")
    os.chdir("univariate_plots")
    
for feat in features:
    skew=df[feat].skew()
    sns.distplot(df[feat],kde=False,label="Skew=%.3f"%(skew),bins=30)
    plt.legend()
    plt.grid(True)
    plt.savefig("{}.jpg".format(feat))
    plt.show()
os.chdir("../")

# 查看离群值
# 使用 Turkey's method 
# http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/

# Detect observations with more than one outlier

# 定义离群值查看函数

def outlier_hunt(df):
    '''
    df: pandas 中的 dataframe 数据格式
    函数返回的是超过两个离群值得indices列表
    '''
    outlier_indices=[]
    
    # iterate over features (columns)
    
    for col in df.columns.tolist():
        # 1st quartile (25%)
        Q1=np.percentile(df[col],25)
        # 3rd quartile (75%)
        Q3=np.percentile(df[col],75)
        
        # Interquartile range (IQR)
        
        IQR=Q3-Q1
        
        
        # outlier step
        
        outlier_step=1.5*IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col=df[(df[col] < Q1-outlier_step) | (df[col] > Q3+outlier_step)].index
        
        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices=Counter(outlier_indices)
    multiple_outliers=list(k for k,v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
    
# 获取离群值的个数
    
print("数据集中的离群值个数如下")
print("The dataset contains %d observations with more than 2 outliers"%(len(outlier_hunt(df[features]))))

# boxplots

if os.path.exists("boxplots"):
    os.chdir("boxplots")
else:
    os.mkdir("boxplots")
    os.chdir("boxplots")

plt.figure(figsize=(8,6))
sns.boxplot(df[features])
plt.savefig("boxplots.jpg",dpi=300)
plt.show()

os.chdir("../")

# Multivariate plots
# 查看变量之间的相关性，考察相互之间的relation
# 多变量绘图
# pairplot

if os.path.exists("multivariate_plots"):
    os.chdir("multivariate_plots")
else:
    os.mkdir("multivariate_plots")
    os.chdir("multivariate_plots")

plt.figure(figsize=(8,8))
sns.pairplot(df[features],palette="coolwarm")
plt.savefig("multivariate_plots.jpg",dpi=300)
plt.show()

os.chdir("../")

# 使用热图进行相关性分析

if os.path.exists("heat_map"):
    os.chdir("heat_map")
else:
    os.mkdir("heat_map")
    os.chdir("heat_map")
    
corr=df[features].corr()
plt.figure(figsize=(16,16))
sns.heatmap(corr,cbar=True,square=True,annot=True,fmt=".2f",annot_kws={"size":15},
            xticklabels=features,yticklabels=features,alpha=0.7,cmap="coolwarm")
plt.savefig("heatmap.jpg",dpi=300)
plt.show()

os.chdir("../")

# dataframe 的数据信息

print(df.info())

# Hunting and removing multiple outliers
# 将离群值剔除

outlier_indices=outlier_hunt(df[features])
df=df.drop(outlier_indices).reset_index(drop=True)

# 获取离群值剔除之后的数据集

print(df.shape)

if os.path.exists("univariate_plots_outlier"):
    os.chdir("univariate_plots_outlier")
else:
    os.mkdir("univariate_plots_outlier")
    os.chdir("univariate_plots_outlier")
    
for feat in features:
    skew=df[feat].skew()
    sns.distplot(df[feat],kde=False,label="Skew=%.3f"%(skew),bins=30)
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("{}.jpg".format(feat))
    plt.show()
    
os.chdir("../")

# 统计去除outlier之后的类型个数

print("统计去除outlier之后的类型个数")
print(df["Type"].value_counts())
print("\n")

# 查看类型分布数据

sns.countplot(df["Type"])
plt.savefig("type_distribution_no_outlier.jpg",dpi=300)
plt.show

# 拆分数据集

# 将X定义成 features， y定义成 labels

X=df[features]
y=df["Type"]

# 设定一个随机数发生子和train/test 分隔比例

seed=7
test_size=0.2

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=seed)

# Data transformation
# 使用 box-cox transform 方法进行数据的归一化

features_boxcox=[]

for feature in features:
    bc_transformed,_=boxcox(df[feature]+1) # shift by 1 to avoid computing log of negative values
    features_boxcox.append(bc_transformed)
    
features_boxcox=np.column_stack(features_boxcox)
df_bc=pd.DataFrame(data=features_boxcox,columns=features)
df_bc["Type"]=df["Type"]

print("输出normalize操作之后的数据描述")
print(df_bc.describe())
print("\n")

# 基于normalize的数据进行绘图对比操作

if os.path.exists("normalize"):
    os.chdir("normalize")
else:
    os.mkdir("normalize")
    os.chdir("normalize")
    
for feature in features:
    fig,ax=plt.subplots(1,2,figsize=(7,3.5))
    ax[0].hist(df[feature],color="blue",bins=30,alpha=0.3,label="Skew = %s"%
      (str(round(df[feature].skew(),3))))
    ax[0].set_title(str(feature))
    ax[0].legend(loc=0)
    ax[1].hist(df_bc[feature],color="red",bins=30,alpha=0.3,label="Skew = %s"%
      (str(round(df_bc[feature].skew(),3))))
    ax[1].set_title(str(feature)+"after a Box-Cox transformation")
    ax[1].legend(loc=0)
    plt.savefig("{}-Box-Cox-transformation.jpg",dpi=300)
    plt.show()
    
os.chdir("../")

# 检查 skew 在经过 box-cox transformation 之后是否接近于0.0

for feature in features:
    delta=np.abs(df_bc[feature].skew()/df[feature].skew())
    if delta < 1.0:
        print("Feature %s is less skewed after a Box-Cox transform"%(feature))
    else:
        print("Feature %s is more skewed after a Box-Cox transform"%(feature))
        

# 下面是使用评估算法了
# Evaluate Algorithms

#降维

# XGBoost
# 先是使用 XGBoost 算法进行重要性分析操作,评估type的重要性
        
model_importances=XGBClassifier()
start=time()
model_importances.fit(X_train,y_train)
print("Elapsed time to train XGBoost %.3f seconds"%(time()-start))
print("\n")

if os.path.exists("xgboost"):
    os.chdir("xgboost")
else:
    os.mkdir("xgboost")
    os.chdir("xgboost")
    
plot_importance(model_importances)
plt.savefig("model_importances.jpg",dpi=300)
plt.show()

os.chdir("../")

# PCA
# 属于无监督学习算法
# 目的：decorrelate the ones that are linearly dependent and then let's 
# plot the cumulative explained variance.
# 简言之就是将具有线性相关性的特征剔除掉

pca=PCA(random_state=seed)
pca.fit(X_train)
var_exp=pca.explained_variance_ratio_
cum_var_exp=np.cumsum(var_exp)

if os.path.exists("pca"):
    os.chdir("pca")
else:
    os.mkdir("pca")
    os.chdir("pca")
    
plt.figure(figsize=(8,6))
plt.bar(range(1,len(cum_var_exp)+1),var_exp,align="center",label="individual variance\
        explained",alpha=0.7)
plt.step(range(1,len(cum_var_exp)+1),cum_var_exp,where="mid",label="cumulative variance explained",\
         color="red")
plt.ylabel("Explained variance ratio")
plt.xlabel("Principal components")
plt.legend(loc="center right")
plt.savefig("pca.jpg",dpi=300)
plt.show()

os.chdir("../")

# cumulative variance explained

for i,sum in enumerate(cum_var_exp):
    print("PC"+str(i+1),"Cumulative variance:%.3f% %"%(cum_var_exp[i]*100))
    
# 开始对比算法
    
# 10-fold cross-validation
# metric: classification accuracy
# 使用 pipeline
    
# 定义通用的超参数
    
n_components=5
pipelines=[]
n_estimators=200

# SVC

pipelines.append(("SVC",Pipeline([("sc",StandardScaler()),
                                  ("SVC",SVC(random_state=seed))])))
    
# KNN
    
pipelines.append(("KNN",Pipeline([("sc",StandardScaler()),
                                  ("KNN",KNeighborsClassifier())])))
    
# RandomForest
    
pipelines.append(("RF",Pipeline([("sc",StandardScaler()),
                                 ("RF",RandomForestClassifier(random_state=seed,
                                                              n_estimators=n_estimators))])))
    
# Adaboost算法
    
pipelines.append(("Ada",Pipeline([("sc",StandardScaler()),
                                  ("Ada",AdaBoostClassifier(random_state=seed,n_estimators=n_estimators))])))

# ET算法
    
pipelines.append(("ET",Pipeline([("sc",StandardScaler()),
                                 ("ET",ExtraTreesClassifier(random_state=seed,
                                                            n_estimators=n_estimators))])))
    
# GB算法
    
pipelines.append(("GB",Pipeline([("sc",StandardScaler()),
                                 ("GB",GradientBoostingClassifier(random_state=seed))])))

# LR分类算法

pipelines.append(("LR",Pipeline([("sc",StandardScaler()),
                                 ("LR",LogisticRegression(random_state=seed))])))

results,names,times=[],[],[]

num_folds=10
scoring="accuracy"

for name,model in pipelines:
    start=time()
    Kfold=StratifiedKFold(n_splits=num_folds,random_state=seed)
    cv_results=cross_val_score(model,X_train,y_train,cv=Kfold,
                               scoring=scoring,n_jobs=1)
    t_elapsed=time()-start
    results.append(cv_results)
    names.append(name)
    times.append(t_elapsed)
    
    msg="%s:%f (+/- %f) performed in %f seconds"%(name,100*cv_results.mean(),
                100*cv_results.std(),t_elapsed)
    print(msg)
    
if os.path.exists("Algorithms"):
    os.chdir("Algorithms")
else:
    os.mkdir("Algorithms")
    os.chdir("Algorithms")
    
fig=plt.figure(figsize=(12,8))
fig.suptitle("Algorithms comparsion")
ax=fig.add_subplot(1,1,1)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.savefig("Algorithms.jpg",dpi=300)
plt.show()

# 输出结果表明 RF 的计算结果是最合适的
# 下面针对 RF 进行参数调整

pipe_rfc=Pipeline([("scl",StandardScaler()),("rfc",RandomForestClassifier(
        random_state=seed,n_jobs=1))])
# 设定 grid search的参数

param_grid_rfc=[{
        "rfc__n_estimators":[100,200,300,400], # 评估器的个数
        "rfc__max_features":[0.05,0.1], # maximum features used at each split
        "rfc__max_depth":[None,5], # 树的最大深度
        "rfc__min_samples_split":[0.005,0.01],# mininal samples in leafs
        }]
    
# Use 10 fold CV

kfold=StratifiedKFold(n_splits=num_folds,random_state=seed)
grid_rfc = GridSearchCV(pipe_rfc, param_grid= param_grid_rfc, cv=kfold, scoring=scoring, verbose= 1, n_jobs=1)

#Fit the pipeline
start = time()
grid_rfc = grid_rfc.fit(X_train, y_train)
end = time()

print("RFC grid search took %.3f seconds" %(end-start))

# Best score and best parameters
print('-------Best score----------')
print(grid_rfc.best_score_ * 100.0)
print('-------Best params----------')
print(grid_rfc.best_params_)

# Diagnose the performance of the best algorithms
# 绘制学习和验证曲线

# Let's define some utility functions to plot the learning & validation curves

def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(train_sizes,train_mean + train_std,
                    train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(train_sizes, test_mean, label='test score', color='red',marker='o')
    plt.fill_between(train_sizes,test_mean + test_std, test_mean - test_std , color='red', alpha=alpha)
    plt.title(title)
    plt.xlabel('Number of training points')
    plt.ylabel('Accuracy')
    plt.grid(ls='--')
    plt.legend(loc='best')
    plt.savefig("learning_curve.jpg",dpi=300)
    plt.show()    
    
def plot_validation_curve(param_range, train_scores, test_scores, title, alpha=0.1):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(param_range,train_mean + train_std,
                    train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(param_range, test_mean, label='test score', color='red', marker='o')
    plt.fill_between(param_range,test_mean + test_std, test_mean - test_std , color='red', alpha=alpha)
    plt.title(title)
    plt.grid(ls='--')
    plt.xlabel('Parameter value')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig("validation_curve.jpg",dpi=300)
    plt.show()    
    
if os.path.exists("learning_validation"):
    os.chdir("learning_validation")
else:
    os.mkdir("learning_validation")
    os.chdir("learning_validation")
    
plt.figure(figsize=(9,6))

train_sizes, train_scores, test_scores = learning_curve(estimator= grid_rfc.best_estimator_ , 
                                                        X= X_train, y = y_train, 
                                                        train_sizes=np.arange(0.1,1.1,0.1), 
                                                        cv= 10,
                                                        scoring='accuracy', 
                                                        n_jobs= 1)

plot_learning_curve(train_sizes, train_scores, 
                    test_scores, title='Learning curve for RFC')

os.chdir("../")


    
      








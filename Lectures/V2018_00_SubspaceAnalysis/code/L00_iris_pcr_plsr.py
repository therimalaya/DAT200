# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 11:52:48 2018

@author: olive
"""

# Import neede modules
import hoggorm as ho
import hoggormplot as hopl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from random import sample



# Load data and extract information
#df = pd.read_table("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", sep=',')
df = pd.read_table("../data/iris_data.txt", sep=',')
df['class'] = df['class'].str.replace('Iris-', '')

classNames = list(df['class'])
XvarNames = list(df.columns[0:4])
YvarName = list(df.columns[4:])
data = df.values


# Compute descriptive statistics
descrStats = df.describe()
#df.plot.box()


# Access data specific to each class
unique_class = list(df['class'].unique())

setosa_df = df[df['class'] == 'setosa']
versicolor_df = df[df['class'] == 'versicolor']
virginica_df = df[df['class'] == 'virginica']

#data_iseto.plot.box(title='Iris setosa')
#data_ivers.plot.box(title='Iris-versicolour')
#data_ivirg.plot.box(title='Iris-virginica')
#
#df.boxplot(by='class')


#pd.plotting.scatter_matrix(df)
# Set up the matplotlib figure
#f, ax = plt.subplots(figsize=(12, 9))

#sns.pairplot(df, hue='class', 
#             markers=["o", "s", "D"],
#             kind='reg')
#
sns.pairplot(df, hue='class', 
             markers=["o", "s", "D"])


# Split data into training and test set
def sample_df(df, train_prop):
  df = df.reset_index(drop=True)
  len_df = int(len(df))
  range_df = range(len_df)
  sample_size = int(np.ceil(len_df * train_prop))
  mask = df.index.isin(sample(range_df, sample_size))
  Train = df.iloc[mask]
  Test = df.iloc[~mask]
  return Train, Test

setosa_train, setosa_test = sample_df(setosa_df, 0.6)
versicolor_train, versicolor_test = sample_df(versicolor_df, 0.6)
virginica_train, virginica_test = sample_df(virginica_df, 0.6)

for name in dir():
    if name.endswith(('train', 'test', 'idx')):
        del globals()[name]

train_df = pd.concat([setosa_train, versicolor_train, virginica_train]).reset_index(drop=True)
test_df = pd.concat([setosa_test, versicolor_test, virginica_test]).reset_index(drop=True)

#train = pd.concat([train_iseto, train_ivers, train_ivirg])
#trainData = train[XvarNames].values
#
#test_iseto = data_iseto.loc[35:49, :]
#test_ivers = data_ivers.loc[85:99, :]
#test_ivirg = data_ivirg.loc[135:149, :]
#test = pd.concat([test_iseto, test_ivers, test_ivirg])
#testData = test[XvarNames].values


# Construct dummy matrix for response (home brew solution)
dummyY = pd.get_dummies(train_df['class'])
#dummyY = np.zeros([np.shape(train)[0], 3])
#
#
#for ind, obj in enumerate(train.index):
#    #print('DF index', ind, ' -- DF row lable', train.index[ind], train.iloc[ind]['class'])
#    if train.iloc[ind]['class'] == 'Iris-setosa':
#        dummyY[ind, 0] = 1
#    elif train.iloc[ind]['class'] == 'Iris-versicolor':
#        dummyY[ind, 1] = 1
#    elif train.iloc[ind]['class'] == 'Iris-virginica':
#        dummyY[ind, 2] = 1




# Compute PCR model
print('PCR **************************************')        
        
modelPCR = ho.nipalsPCR(arrX=train_df[XvarNames].values,
                     arrY=dummyY, 
                     Xstand=True, Ystand=False,
                     cvType=['loo'])


hopl.plot(modelPCR, comp=[1,2], 
          plots=[1, 2, 4, 6],
          objNames = list(train_df['class']),
          XvarNames = XvarNames,
          YvarNames = ['Setosa', 'Versicolor', 'Virginica'])

hopl.explainedVariance(modelPCR, validated=[True], which=['X'])

predRes = modelPCR.Y_predict(test_df[XvarNames].values , numComp=2)




## Compute PLSR model
#print('PLS2 **************************************')
#
#modelPLS2 = ho.nipalsPLS2(arrX=trainData,
#                          arrY=dummyY, 
#                          Xstand=True, Ystand=False,
#                          cvType=['loo'])
#
#
#hopl.plot(modelPLS2, comp=[1,2], 
#          plots=[1, 2, 3, 4, 6],
#          objNames = list(train['class']),
#          XvarNames = XvarNames,
#          YvarNames = ['Setosa', 'Versicolor', 'Virginica'])
#
#predRes = modelPLS2.Y_predict(testData , numComp=2)





re = predRes.max(axis=1)

classRes = np.zeros([np.shape(predRes)[0], np.shape(predRes)[1]])
for col in range(np.shape(predRes)[1]):
    #print(col)
    
    for row in range(np.shape(predRes)[0]):
        #print(col, '--', row)
        
        if predRes[row, col] == re[row]:
            classRes[row, col] = 1


correctSetosa = np.sum(classRes[0:15, 0])
correctVersicolor = np.sum(classRes[15:30, 1])
correctVirginica = np.sum(classRes[30:45, 2])

corrSetosaPerc = correctSetosa / 15 * 100
corrVersicolorPerc = correctSetosa / 15 * 100
corrVirginicaPerc = correctSetosa / 15 * 100

totalPerc = np.sum(correctSetosa + correctVersicolor + correctVirginica) / 45 * 100
print(totalPerc)
print(np.sum(correctSetosa + correctVersicolor + correctVirginica))


zz = np.argmax(predRes, axis=1)



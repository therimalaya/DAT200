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



# Load data and extract information
#df = pd.read_table("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", sep=',')
df = pd.read_table("iris_data.txt", sep=',')
classNames = list(df['class'])
XvarNames = list(df.columns[0:4])
YvarName = list(df.columns[4:])
data = df.values


# Compute descriptive statistics
descrStats = df.describe()
#df.plot.box()


# Access data specific to each class
uniqueCl = list(df['class'].unique())

data_iseto = df[df['class'] == 'Iris-setosa']
data_ivers = df[df['class'] == 'Iris-versicolor']
data_ivirg = df[df['class'] == 'Iris-virginica']

#desSt_iseto = data_iseto.describe()
#desSt_ivers = data_ivers.describe()
#desSt_ivirg = data_ivirg.describe()


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
train_iseto = data_iseto.loc[0:34, :]
train_ivers = data_ivers.loc[50:84, :]
train_ivirg = data_ivirg.loc[100:134, :]
train = pd.concat([train_iseto, train_ivers, train_ivirg])
trainData = train[XvarNames].values

test_iseto = data_iseto.loc[35:49, :]
test_ivers = data_ivers.loc[85:99, :]
test_ivirg = data_ivirg.loc[135:149, :]
test = pd.concat([test_iseto, test_ivers, test_ivirg])
testData = test[XvarNames].values


# Construct dummy matrix for response (home brew solution)
dummyY = np.zeros([np.shape(train)[0], 3])


for ind, obj in enumerate(train.index):
    #print('DF index', ind, ' -- DF row lable', train.index[ind], train.iloc[ind]['class'])
    if train.iloc[ind]['class'] == 'Iris-setosa':
        dummyY[ind, 0] = 1
    elif train.iloc[ind]['class'] == 'Iris-versicolor':
        dummyY[ind, 1] = 1
    elif train.iloc[ind]['class'] == 'Iris-virginica':
        dummyY[ind, 2] = 1




# Compute PCR model
print('PCR **************************************')        
        
modelPCR = ho.nipalsPCR(arrX=trainData,
                     arrY=dummyY, 
                     Xstand=True, Ystand=False,
                     cvType=['loo'])


hopl.plot(modelPCR, comp=[1,2], 
          plots=[1, 2, 3, 4, 6],
          objNames = list(train['class']),
          XvarNames = XvarNames,
          YvarNames = ['Setosa', 'Versicolor', 'Virginica'])

hopl.explainedVariance(modelPCR, validated=[True], which=['X'])

predRes = modelPCR.Y_predict(testData , numComp=2)




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



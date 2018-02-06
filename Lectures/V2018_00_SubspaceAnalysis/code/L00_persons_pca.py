# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:28:42 2018

@author: olive
"""

# Import neede modules
import hoggorm as ho
import hoggormplot as hopl
import pandas as pd


# Load data and extract information
persons_df = pd.read_excel("Persons.xlsx", sep='\t', index_col=0)
persNames = list(persons_df.index)
charNames = list(persons_df.columns)
persData = persons_df.values

#persons_df['Height (cm)'] = persons_df['Height (cm)'] * 10000

# Compute PCA model
pers_model = ho.nipalsPCA(arrX=persData,
                          numComp=8,
                          Xstand=True,
                          cvType=["loo"])

hopl.plot(pers_model, comp=[1, 2], 
          plots=[1, 2, 6], 
          objNames=persNames,
          XvarNames=charNames)



cumCalExplVar = pers_model.X_cumCalExplVar()




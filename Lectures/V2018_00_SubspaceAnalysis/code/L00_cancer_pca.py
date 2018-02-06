# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:31:53 2018

@author: olive
"""


# Import neede modules
import hoggorm as ho
import hoggormplot as hopl
import pandas as pd


# Load data and extract information
cancer_men_df = pd.read_table("Cancer_men_perc.txt", sep='\t', index_col=0)
countryNames = list(cancer_men_df.index)
cancerNames = list(cancer_men_df.columns)
cancerDataMen = cancer_men_df.values


# Compute PCA model
cm_model = ho.nipalsPCA(arrX=cancerDataMen,
                        numComp=8,
                        Xstand=False,
                        cvType=["loo"])

hopl.plot(cm_model, comp=[1, 2], 
          plots=[1, 2, 3, 4, 6], 
          objNames=countryNames,
          XvarNames=cancerNames)


cumCalExplVar = cm_model.X_cumCalExplVar()
calExplVar = cm_model.X_calExplVar()









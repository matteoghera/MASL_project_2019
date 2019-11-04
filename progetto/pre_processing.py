#librerie di sistema per gestire i percorsi
import os, sys, random, io, urllib
from datetime import datetime


#importazione delle librerie di data science
import pandas as pd
import numpy as np
import seaborn as sns
import  torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import Image, display


#settaggio per lo stile grafico
sns.set_style('darkgrid')


#importazione dati
url = 'https://raw.githubusercontent.com/GitiHubi/deepAI/master/data/fraud_dataset_v2.csv'

df = pd.read_csv(url)
print(df.head())
print(df.dtypes)
print("La lunghezza del dataset è  {}".format(df.shape))
print(df.isnull().sum())
#non ci sono missing values, si procede con un attività riassuntiva dei dati
print(df.describe())
sns.heatmap(data=df.corr())
plt.show()
sns.scatterplot(x="DMBTR", y="WRBTR",data=df)
plt.show()
#vediamo che contengono le singole colonne
categoric_variable = list(set(df.columns)-set(['BELNR ', 'WRBTR', 'DMBTR']))
for c_var in categoric_variable:
    sns.countplot(x=c_var, data=df)
    plt.show()
    #print(df[c_var].value_counts())
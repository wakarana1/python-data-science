import pandas as pd
import numpy as np # so you can use np.log()
import matplotlib.pyplot as plt

from scipy import stats

%matplotlib inline
df = pd.read_csv('kc_house_data.csv')

df.info() # gives snapshop of the data
df['zipcode'].nunique() # number of unique, similar to sql distinct
df['zipcode'].unique() # gives unique values
df['zipcode'].value_counts() # similar to sql grou_by
df.describe()

# MEAN MEDIAN MODE
df['zipcode'].mean()
df['zipcode'].median()
df['zipcode'].mode()


df.describe() # shows descriptive stats
df[df.bedrooms==33] # looks for bedroom with 33 rooms


df.sqft_living.plot() # plots graph of sqft_living key

pred_price = stats.linregress(df.sqft_living,df.price) # gets the slope and y intercept through linear regression
# LinregressResult(slope=280.62356789744831, intercept=-43580.743094474077, rvalue=0.70203505461180027, pvalue=0.0, stderr=1.9363985519989133)

def predict(x):
  return pred_price.slope * x + pred_price.intercept

predict(5400) #finding price of 5400 sqft house
# 1471786.5235517467

df[df['sqft_living']==5400]
#1210000.0

plt.hist(np.log(df.price), bins=200)

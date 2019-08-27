



import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import matplotlib.pyplot as plt
import io
from sklearn import tree
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from __future__ import division
from __future__ import division
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from collections import Counter
os.chdir('/home/prudhvi/Documents')

housing_df = pd.read_csv('HousingMarket.csv' , sep='\t',error_bad_lines=False)


reg_wise_med = housing_df.groupby('RegionTypeDesc')['Median House Value']

sample = housing_df[housing_df['fixed']== 103500]

sample['Date'].unique()

reg_wise_med = housing_df.groupby('fixed')['Date'].unique()


print confusion_matrix(housing_df.iloc[:,3],housing_df.iloc[:,13])
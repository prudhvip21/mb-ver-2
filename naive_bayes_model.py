#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:59:50 2017

@author: prudhvi
"""


import os
import time

#os.chdir('/root/mb/market_basket_data')
os.chdir('/home/prudhvi/Documents/market_basket_data') 


from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score, confusion_matrix
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)  


start_time  = time.time()

import time
start_time = time.time()


order_products_train_df = pd.read_csv("order_products__train.csv")
order_products_prior_df = pd.read_csv("order_products__prior.csv") 
orders_df = pd.read_csv("orders.csv")
products_df = pd.read_csv("products.csv")
aisles_df = pd.read_csv("aisles.csv")
departments_df = pd.read_csv("departments.csv")


print time.time() - start_time

print ("--- %s seconds ---" % (time.time() - start_time))




""" Simple Naive Bayes Model """ 



""" Creating overall probability of purchase for the each product_id""" 

count_df = order_products_prior_df.groupby("product_id")["reordered"].aggregate('count') 
  # count_df will have counts of number of times order is purchased
count_df.name = 'Total Count'
sum_df = order_products_prior_df.groupby("product_id")["reordered"].aggregate('sum') 
sum_df.name = 'Reorders count'
   # sum_df will have number of times the product is re-ordered.
df = pd.concat([count_df,sum_df],axis= 1).reset_index()
df['Reorder_ratio'] = df['Reorders count']/df['Total Count']
# re-order ratio is counted for each product and saved as table based on all prior orders

""" creating probabilities for each user_id and Product id """ 

order_products_prior_df = pd.merge(order_products_prior_df,orders_df,on = 'order_id' , how = 'left') 
  # extracting information of other columns likes user id for prior orders
overall_sum_df = order_products_prior_df.groupby(["user_id","product_id"])['reordered'].aggregate('sum') 
  # for each combination of user id and product id, calculating the number of times it is reordeered. 
overall_sum_df.name = "Reorders Count"
overall_count_df = order_products_prior_df.groupby(["user_id","product_id"])['reordered'].aggregate('count')
overall_count_df.name = "Total Count" 
   # for each user id and product id combination calculating , total number of times it is ordered
df2 = pd.concat([overall_sum_df,overall_count_df],axis = 1).reset_index()
df2['ratio'] = df2['Reorders Count']/df2['Total Count']
# data frame for probabilities of product_id, user id combination p(O/R) 


""" Calculating final proababilities after multiplication """ 

# merging the prudct_id , user_id combo probabilities with product_id probabilites 
df2 = pd.merge(df2,df.iloc[:,[0,3]],on = 'product_id' , how = 'left')
df2.rename(columns = {'ratio' : 'User_Reorder_Ratio'} , inplace = True)
df2.rename(columns = {'Reorder_ratio' : 'Overall_Reorder_Ratio'} , inplace = True)
df2['final_prob'] = df2['Overall_Reorder_Ratio']*df2['User_Reorder_Ratio']
# calculating the final probability for each product id and user id combination 

df2['final_prob'] = np.where(df2['final_prob']==0, df2['Overall_Reorder_Ratio'] , df2['final_prob'])
# if some user_id, product_id probability is zero, overal prabibility is taken
df2['predicted'] = np.where(df2['final_prob']>0.28,1,0)
# fixing the cut-off probability and assigning 0 or 1
grouped = df2[df2['predicted']==1] 
# subgrouping only reordered predicted items 
def spaces(k):
    s = ""
    for item in k :
        s = s + str(item) + " " 
    return s  
# Spaces is a function to take each group and convert on column into a list and paste it to corresponding row. 
grouped = grouped.groupby(['user_id']).apply(lambda x : spaces(x['product_id'].tolist())).reset_index() 
# takes each user_id , finds all the product_ids predicted and puts them into list and pastes it to the corresponding row  
df_test_orders = orders_df[orders_df['eval_set']=='test']
# taking test orders only 
df_test_orders = df_test_orders.drop(df_test_orders.columns[[2,3,4,5,6]], axis=1) 
final_test_sub = pd.merge(df_test_orders,grouped, on = 'user_id', how = 'left')
# final_test_submissions will have only test orders and correspoinding predicted items. 
final_test_sub.rename(columns = {0 : 'Products'} , inplace = True)
del final_test_sub['user_id']
del final_test_sub['Predicted_Products']
# removing unnexessary attributes
final_test_sub.to_csv( path_or_buf ="~/sub.csv", header = True )
# writing the file to CSV

""" Train Data using for model """ 

df_train_orders = orders_df[orders_df['eval_set']=='train']
# taking test orders only 
df_train_orders = df_train_orders.drop(df_train_orders.columns[[2,3,4,5,6]], axis=1) 
final_train_pred = pd.merge(df_train_orders,grouped,on = 'user_id' , how = 'left')
final_train_pred.rename(columns = {0 : 'Products'} , inplace = True)

train_actual_df = order_products_train_df[order_products_train_df['reordered']==1]

train_grouped = train_actual_df.groupby(['order_id']).apply(lambda x : spaces(x['product_id'].tolist())).reset_index()  
train_grouped.rename(columns = {0 : 'Actual Products'} , inplace = True)

train_grouped = pd.merge(train_grouped,final_train_pred,on = 'order_id' , how = 'left')

from __future__ import division 
def cal_metrics(s):
    try : 
        
        act = s['Actual Products'].split(' ')
        act = [ item for item in act if item != '']
        pred = s['Products'].split(' ')
        pred = [ item for item in pred if item != ''] 
        TP = len(list(set(act).intersection(pred)))
        PP = len(pred)
        AP = len(act)
        score = (2*TP) / (AP + PP)
        return pd.Series({'F1' : score})
    except : 
         k = 1 
         
train_grouped = train_grouped.merge(train_grouped.apply(cal_metrics, axis=1), left_index=True, right_index=True)

probs = np.arange(0.1,0.3,0.01)
for item in probs :
    
    df2['predicted'] = np.where(df2['final_prob']>item,1,0)
    grouped = df2[df2['predicted']==1] 
    grouped = grouped.groupby(['user_id']).apply(lambda x : spaces(x['product_id'].tolist())).reset_index() 
    grouped.rename(columns = {0 : 'Products'} , inplace = True)
    final_train_pred = pd.merge(df_train_orders,grouped,on = 'user_id' , how = 'left') 
    train_grouped = train_actual_df.groupby(['order_id']).apply(lambda x : spaces(x['product_id'].tolist())).reset_index()  
    train_grouped.rename(columns = {0 : 'Actual Products'} , inplace = True)
    train_grouped = pd.merge(train_grouped,final_train_pred,on = 'order_id' , how = 'left')
    train_grouped = train_grouped.merge(train_grouped.apply(cal_metrics, axis=1), left_index=True, right_index=True)
    print "For Probability :", item ,"F1 Score is ", np.mean(train_grouped['F1'])
    del train_grouped
    
    


"""
del df2['Reorders Count'],df2['Total Count']
df_final = pd.merge(df_submissions, df2 , on = ['user_id', 'product_id'],how = 'outer')
# df_orders = pd.merge(df_final,df_submissions, on = 'order_id' , how = 'left')
del df_final['Overall_Reorder_Ratio'],df_final['User_Reorder_Ratio'] 
df_submissions['final_prob'] = df_submissions['Overall_Reorder_Ratio']*df_submissions['User_Reorder_Ratio']

for index, row in df_submissions.iterrows():
    if np.isnan(row['final_prob']) :
        row['final_prob'] = row['Overall_Reorder_Ratio']
        print index 

df_submissions['final_prob'] = np.where(np.isnan(df_submissions['final_prob']), df_submissions['Overall_Reorder_Ratio'] , df_submissions['final_prob'])
df_submissions['predicted'] = np.where(df_submissions['final_prob']>0.4,1,0)
print accuracy_score(df_submissions['reordered'],df_submissions['predicted'])
print confusion_matrix(df_submissions['reordered'],df_submissions['predicted']) 
print f1_score(df_submissions['reordered'],df_submissions['predicted']) 
df3 = df_submissions.groupby(['order_id'])['reordered','predicted'].apply(get_f1) 
df3 = df_submissions.groupby(['order_id'])['reordered','predicted'].apply(lambda x : get_f1(x['reordered'],x['predicted']))

def get_f1(x,y):
    return f1_score(x,y) 
df_test = orders_df[orders_df['eval_set']=='test']


"""  


"""
df_test_orders = orders_df[orders_df['eval_set']=='test']
# taking test orders only 
df_test_orders = df_test_orders.drop(df_test_orders.columns[[2,3,4,5,6]], axis=1)
# removing not required columns of test. 
df_test_orders['Predicted_Products'] = ""
count = 1
for index,row in df_test_orders.iterrows():
    user_id = row['user_id']
    sample_df = df2[df2['user_id']==user_id]
    sample_df = sample_df[sample_df['predicted']==1]
    k=  sample_df['product_id'].tolist()
    s = ""
    for item in k :
        s = s + str(item) + " " 
    df_test_orders['Predicted_Products'][index] = s 
    print count 
    count = count + 1

"""  



""" 
# train data frame modelling 

df_submissions = pd.merge(order_products_train_df,orders_df, on = 'order_id' , how ='left')
  # Getting train order seperately into one data frame 
del df_submissions['add_to_cart_order'],df_submissions['eval_set'],df_submissions['order_number'],df_submissions['order_dow'],df_submissions['order_hour_of_day'], df_submissions['days_since_prior_order']
df_submissions = pd.merge(df_submissions,df.iloc[:,[0,3]],on = 'product_id' ,how = 'left')
df_submissions.rename(columns = {'Reorder_ratio' : 'Overall_Reorder_Ratio'} , inplace = True )
df_submissions = pd.merge(df_submissions, df2.iloc[:,[0,1,4]], on = ['product_id','user_id'], how = 'left') 
df_submissions.rename(columns = {'ratio' : 'User_Reorder_Ratio'} , inplace = True)

"""  
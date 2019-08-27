
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from collections import Counter
print('loading train')

train = pd.read_csv('order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading orders')
orders = pd.read_csv('orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,

        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

print('loading products')
products = pd.read_csv('products.csv', dtype={
        'product_id': np.uint16,
        'product_name': 'category',
        'aisle_id': np.uint8,
        'department_id': np.uint8},
        usecols=['product_id','product_name','aisle_id', 'department_id'])

departments = pd.read_csv('departments.csv')
aisles = pd.read_csv('aisles.csv')

priors = priors.merge(products, on = 'product_id')
priors = priors.merge(orders, on = 'order_id')
train = train.merge(orders, on = 'order_id')

orders.set_index('order_id', inplace=True, drop=False)

priors_alcohol = priors[priors.department_id == 5]
priors_pets = priors[priors.department_id == 8]



alcohol_users = priors_alcohol.user_id.unique()
pet_users = priors_pets.user_id.unique()

priors = priors[priors['user_id'].isin(alcohol_users)]
train_alcohol = train[train['user_id'].isin(alcohol_users)]

print('computing product f')
prods = pd.DataFrame()
prods['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
products = products.join(prods, on='product_id',rsuffix = '_')
products.set_index('product_id', drop=False, inplace=True)
del prods


print('computing user f')

usr = pd.DataFrame()
usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)
users = pd.DataFrame()
users['total_items'] = priors.groupby('user_id').size().astype(np.int16)
users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)
users = users.join(usr)



print('compute userXproduct f - this is long...')
priors['user_product'] = priors.product_id + priors.user_id * 100000

d= dict()
i = 0
for row in priors.itertuples():
    if i % 10000 == 0: print('order row', i)
    z = row.user_product
    if z not in d:
        d[z] = (1,
                (row.order_number, row.order_id),
                row.add_to_cart_order)
    else:
        d[z] = (d[z][0] + 1,
                max(d[z][1], (row.order_number, row.order_id)),
                d[z][2] + row.add_to_cart_order)
    i = i + 1

print('to dataframe (less memory)')
userXproduct = pd.DataFrame.from_dict(d, orient='index')
del d

userXproduct = userXproduct.reset_index()
userXproduct.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart']
userXproduct.nb_orders = userXproduct.nb_orders.astype(np.int16)
userXproduct.last_order_id = userXproduct.last_order_id.map(lambda x: x[1]).astype(np.int32)
userXproduct.sum_pos_in_cart = userXproduct.sum_pos_in_cart.astype(np.int16)


""" orders """

orders_alcohol = orders[orders.user_id.isin(list(alcohol_users))]
orders_train = orders_alcohol[orders_alcohol['eval_set']=='train']

train = train_alcohol

train.set_index(['order_id', 'product_id'], inplace=True, drop=False)


def features(selected_orders, labels_given=False):
    print('build candidate list')
    order_list = []
    product_list = []
    labels = []
    i = 0
    print len(selected_orders)
    for row in selected_orders.itertuples():
        # print i
        i += 1
        if i % 1000 == 0: print('order row', i)
        order_id = row.order_id
        user_id = row.user_id
        user_products = users.all_products[user_id]
        product_list += user_products
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels += [(order_id, product) in train.index for product in user_products]

    print Counter(labels)
    df = pd.DataFrame({'order_id': order_list, 'product_id': product_list}, dtype=np.int32)
    labels = np.array(labels, dtype=np.int8)
    del order_list
    del product_list
    print('user related features')
    df['user_id'] = df.order_id.map(orders.user_id)
    df['user_total_orders'] = df.user_id.map(users.nb_orders)
    df['user_total_items'] = df.user_id.map(users.total_items)
    df['total_distinct_items'] = df.user_id.map(users.total_distinct_items)
    df['user_average_days_between_orders'] = df.user_id.map(users.average_days_between_orders)
    #df['user_average_basket'] = df.user_id.map(users.average_basket)

    print('order related features')
    # df['dow'] = df.order_id.map(orders.order_dow)
    df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)
    df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
    df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders

    print('product related features')
    df['aisle_id'] = df.product_id.map(products.aisle_id)
    df['department_id'] = df.product_id.map(products.department_id)
    df['product_orders'] = df.product_id.map(products.orders).astype(np.int32)
    df['product_reorders'] = df.product_id.map(products.reorders)
    df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)

    print('user_X_product related features')
    df['z'] = df.user_id * 100000 + df.product_id
    df.drop(['user_id'], axis=1, inplace=True)
    #df['inter_feature'] = df.z.map(userXproduct.inter_feature)

    df['UP_orders'] = df.z.map(userXproduct.nb_orders)
    df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
    df['UP_average_pos_in_cart'] = (df.z.map(userXproduct.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
    df['UP_reorder_rate'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(orders.order_number)
    """ 
    df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day - df.UP_last_order_id.map(orders.order_hour_of_day)).map(
        lambda x: min(x, 24 - x)).astype(np.int8)
    """
    # df['UP_same_dow_as_last_order'] = df.UP_last_order_id.map(orders.order_dow) == \
    #                                              df.order_id.map(orders.order_dow)
    #df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)
    print(df.dtypes)
    print(df.memory_usage())
    return (df, labels)

df_train,labels = features(orders_train, labels_given=True)

f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
       'user_average_days_between_orders',
       'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio','product_orders','aisle_id','department_id','product_reorders',
       'product_reorder_rate', 'UP_orders', 'UP_orders_ratio','UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last']

print('formating for lgb')
d_train = lgb.Dataset(df_train[f_to_use],
                      label=labels,
                      categorical_feature=['aisle_id', 'department_id'])

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 96,
    'max_depth': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}
ROUNDS = 100

print('light GBM train :-)')
bst = lgb.train(params, d_train, ROUNDS)

preds = bst.predict(df_train[f_to_use])

df_train['preds'] = preds

TRESHOLD = 0.18

d = dict()

for row in df_train.itertuples():
    if row.preds > TRESHOLD:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

sub = pd.DataFrame.from_dict(d, orient='index')
sub = sub.reset_index()
sub = sub.rename(columns = { 0 : 'Predicted' ,'index' : 'order_id'})

products_orders_df = train_alcohol.groupby(['order_id']).apply(lambda x: x['product_id'].tolist()).reset_index()
products_orders_df = products_orders_df.rename(columns = { 0 : 'Actual'})
sub['Actual'] = sub.order_id.map(products_orders_df['Actual'])

def cal_metrics(s):
    try:
        act = s['Actual']
        #print act
        pred = s['Predicted'].split(' ')
        pred = [int(item) for item in pred if item != '']
        #print pred
        TP = len(list(set(act).intersection(pred)))
        PP = len(pred)
        AP = len(act)
        #print "precision " ,(TP/PP)
        #print "Recall ", (TP / AP)
        score = (2 * TP) / (AP + PP)
        return pd.Series({'F1': score})
    except Exception,e :
        #print e
        pass

len(set(list(sub['order_id'])).intersection(list(products_orders_df['order_id'])))
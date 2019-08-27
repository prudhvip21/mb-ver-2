

from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

print('loading prior')
priors = pd.read_csv('order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

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


""" Counts of different departments and clustering code """


# Generating priors for each user set
priors = priors.merge(products, on = 'product_id')
priors = priors.merge(orders, on = 'order_id')

priors_produce = priors[priors.department_id == 4]
priors_alcohol = priors[priors.department_id == 5]
priors_pets = priors[priors.department_id == 8]
priors_babies = priors[priors.department_id == 18]

# generating users for each group prior set.

all_users = priors.user_id.unique()
produce_users = priors_produce.user_id.unique()
alcohol_users = priors_alcohol.user_id.unique()
pet_users = priors_pets.user_id.unique()
babies_users = priors_babies.user_id.unique()

col = ['alcohol' ,'pet' ,'baby']

# clustering data frame preparation

df = pd.DataFrame(0 ,index = all_users , columns = col)
df = df.reset_index()
df = df.rename(columns = { 'index' : 'user_id'})
df['alcohol'] =df.apply(lambda row: 1 if row['user_id'] in alcohol_users else 0, axis=1)
df['pet'] = df.apply(lambda row: 1 if row['user_id'] in pet_users else 0, axis=1)
df['baby'] = df.apply(lambda row: 1 if row['user_id'] in babies_users else 0, axis=1)

# Days since criter preparation
group = orders.groupby('user_id')['days_since_prior_order'].aggregate('median').reset_index()
group['binned_days_since'] = pd.cut(group.days_since_prior_order, [0, 4, 9, 17, 24, 30], labels=[1, 2, 3, 4, 5])
del group['days_since_prior_order']
df = df.merge(group, on='user_id')
df = df.fillna(value=1)
inertia = []

# Finding optimum clusters
for i in range(2,14) :

    kmeans = KMeans(n_clusters=i, random_state=0).fit(df.ix[:,1:5])
    inertia.append(kmeans.inertia_)
    print "For" , i , "clusters, inertia is" ,kmeans.inertia_



# Kmeans

kmeans = KMeans(n_clusters=4, random_state=0).fit(df.ix[:, 1:5])
df['labels'] = kmeans.labels_

# using decision tree for rules generation from clusters
clf = DecisionTreeClassifier(random_state=0)
clf.fit(df.ix[:, 1:5], kmeans.labels_)

""" Ratio of departments share among fresh produce users """

df_prior_product = pd.merge(left=priors, right=products, how='left').drop('product_id', axis=1)

df_prior_product = df_prior_product.merge(aisles, on='aisle_id')
df_prior_product = df_prior_product.merge(departments, on='department_id')

plt.figure(figsize=(10, 10))
temp_series = df_prior_product['department'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())))
print sum(sizes[:6])
plt.figure(figsize=(20, 8))
sns.barplot(labels, sizes, alpha=0.8, color='C0')
# plt.barplot(sizes, labels=labels, autopct='%1.1f%%', startangle=200)
plt.title("Departments distribution", fontsize=15)
plt.xticks(rotation='vertical', fontsize=12)
plt.show()

all_produce_users = list(priors_produce.user_id.unique())

df_produce_prior = df_prior_product[df_prior_product['user_id'].isin(all_produce_users)]


""" code for changing products names 
import re
products['product_name'] = products.apply(lambda row : re.sub('[^A-Za-z0-9]+', '', row['product_name']),axis = 1)
products.to_csv('products.csv')

"""


""" code for combining priors and train 

priors = pd.concat([priors,train])
priors = priors.merge(products , on='product_id')
priors.to_csv('priors.csv')

"""
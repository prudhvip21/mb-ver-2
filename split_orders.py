


print('loading orders')
orders = pd.read_csv('orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,

        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

k = 0
for i in range(10) :
    index = range(k,k+342108)
    df_name = 'orders_' + str(i) + '.csv'
    df = orders.ix[index]
    df.to_csv(df_name)
    if i == 9 :
        k = 3421083
    else  :
         k = k + 342108



print('loading prior')
priors = pd.read_csv('order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})


k = 0
for i in range(33) :
    index = range(k,k+1000000)
    df_name = 'priors_splits/priors_' + str(i) + '.csv'
    df = priors.ix[index]
    df.to_csv(df_name , index = False)
    k = k + 1000000
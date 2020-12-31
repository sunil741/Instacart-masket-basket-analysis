from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import numpy as np
from sklearn import linear_model
import tensorflow as tf
import os
import json
from sklearn.feature_extraction.text import CountVectorizer





# https://www.tutorialspoint.com/flask

import flask

app = Flask(__name__)





###################################################




@app.route('/')

def hello_world():
    return 'Hello World!'




def prepare_data(x):
  '''function to make a list of products as expected in the competition'''
  return ' '.join(list(x.astype(str)))

def prepare_products_data(x):
  '''function to make a list of products as expected in the competition'''
  return ' *'.join(list(x.astype(str)))


def mean_f1score(X):
  '''this function returns the mean of f1 scores calculated over different orders'''
  f1_scores=[]
  y_true=X.true_labels
  y_pred=X.products
  for i in range(len(y_true)):
    true_products=set(y_true[i].split(' '))
    if(len(true_products)==0):
      f1_scores.append(0.0)
      break
    pred_products=set(y_pred[i].split(' '))
    pr=len(pred_products.intersection(true_products))/len(pred_products)
    re=len(pred_products.intersection(true_products))/len(true_products)
    if(pr+re==0):
      f1_scores.append(0.0)
    else:
      f1_scores.append((2*pr*re)/(pr+re))
  return np.mean(f1_scores)

#https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def memory_decrease_by_column(df):
  '''This function helps to reduce the memory taken by the dataframe by storing the data in data types of best fit'''  
  col_type_dict={'order_id':np.uint32,'user_id':np.uint32,'order_number':np.uint8,
                 'order_dow':np.uint8,'order_hour_of_day':np.uint8,'days_since_prior_order':np.float16,
                 'product_id':np.uint16,'add_to_cart_order':np.uint8,'reordered':np.uint8,
                 'aisle_id':np.uint8,'department_id':np.uint8,'reordered_new':np.uint8,
                 'user_max_ono':np.uint8,'user_sum_reord':np.uint16,'user_reord_prop':np.float16,
                  'user_prod_reord_prop':np.float16,'user_uniqpr':np.uint16,'user_uniqpr_prop':np.float16,
                  'user_order_reord_prop':np.float16,'user_dsp_mean':np.float16,'user_min_order_size':np.uint8,
                  'user_max_order_size':np.uint8,'user_mean_order_size':np.float16,
                  'product_ratios_users_oneshot':np.float16,'product_cart_mean':np.float16,
                  'product_reord_count':np.uint32,'product_reord_prop':np.float16,
                  'prod_uniq_us':np.uint32,'prod_uniq_us_prop':np.float16,
                  'prod_us_reord_prop':np.float16,'user_days_since_product':np.float16,'user_product_hod_mean':np.float16,
                'user_product_dow_mean':np.float16,'user_product_prop':np.float16,
                'user_product_cnt':np.uint8,'user_product_atc_mode_min':np.uint8,
                'user_product_atc_mode_max':np.uint8,'user_product_atc_min':np.uint8,
                'user_product_atc_max':np.uint8,'user_product_atc_mean':np.float16,
                'aisle_reordered':np.uint32,'aisle_reordered_prop':np.float16,
                'dep_reordered':np.uint32,'dep_reordered_prop':np.float16,
                'order_dow_reordered':np.uint32,'order_dow_reordered_prop':np.float16,
                'order_hod_reordered':np.uint32,'order_hod_reordered_prop':np.float16,
                'order_dow_hod_reord_count':np.uint32,'ono_dsp_reord':np.uint32,
                'order_dow_hod_reord_prop':np.float16,'ono_dsp_reord_prop':np.float16,
                'atc_reordered':np.uint32,
                'atc_reordered_prop':np.float16,'product_ordered_today':np.uint8,
                'user_days_since_product_corrected':np.float16}

  for i in df.columns:
    if i!='eval_set':
      df[i]=df[i].astype(col_type_dict[i])
  return df

@app.route('/index')
def index():
	# return ' '.join(os.listdir('.'))
    return flask.render_template('index.html')

@app.route('/index1')
def index1():
    return flask.render_template('index1.html')

@app.route('/arrjson')
def return_ar():
  a=[1,2,3]
  b=np.array(a)
  return jsonify({'b':b})

@app.route('/listjson')
def return_list():
  a=[1,2,3]
  b=np.array(a).tolist()
  return jsonify({'b':b})





@app.route('/predict', methods=['POST'])
def final_fun_1():
  '''This function returns the products that the users might reorder in the given orders'''
  #preparing features
  inp= request.form.to_dict()
  
  X=pd.DataFrame({'order_id':inp['order_id'].split(','),'user_id':inp['user_id'].split(','),'order_dow':inp['order_dow'].split(','),'order_number':inp['order_number'].split(','),
	'order_hour_of_day':inp['order_hour_of_day'].split(','),'days_since_prior_order':inp['days_since_prior_order'].split(',')});
  X=memory_decrease_by_column(X)
  all_info=pd.merge(X,user_all_info[user_all_info.user_id.isin(X.user_id)],on='user_id',how='left')
  all_info=pd.merge(all_info,order_dow_features,on='order_dow',how='left')
  all_info=pd.merge(all_info,order_hod_features,on='order_hour_of_day',how='left')
  all_info=pd.merge(all_info,order_dow_hod_features,on=['order_dow','order_hour_of_day'],how='left')
  all_info=pd.merge(all_info,ono_dsp_features,on=['order_number','days_since_prior_order'],how='left')
  all_info['user_days_since_product_corrected']=all_info['user_days_since_product']+all_info['days_since_prior_order']
  all_info['product_ordered_today']=all_info['user_days_since_product_corrected'].apply(lambda x: 1 if x==0 else 0)
  #data cleaning
  all_info.fillna(0,inplace=True)
  X_test=all_info[train_columns]
  X_test=(X_test-X_train_min)/(X_train_max_min)
  #model evaluation
  best_model=tf.keras.models.load_model('conv_model_f10.3654403235929674')
  pred_test_y=(best_model.predict(X_test,batch_size=1000)>=0.2)
  #output preparation
  all_info['pred_reordered']=pred_test_y
  submission=all_info[all_info.pred_reordered==1][['order_id','product_id']]
  submission=pd.merge(submission,products,on='product_id',how='left')
  submission.columns=['order_id','products','product_name']
  submission=submission.groupby('order_id').agg({'products':prepare_data,'product_name':prepare_products_data}).reset_index()
  submission=pd.merge(X[['order_id']],submission,how='left',on='order_id')
  submission.fillna('None',inplace=True)  
  submission['len']=submission.products.apply(lambda x: 0 if x=='None' else len(x.split(' ')))
  submission['products']=submission.apply(lambda x: x.products+' None' if (x.len==1 or x.len==2) else x.products ,axis=1)
  order_ids=submission.order_id.values.tolist()
  product_ids=submission.products.values.tolist()
  product_names=submission.product_name.values.tolist()
  # order_ids='#'.join([str(i) for i in list(order_ids)])
  return flask.render_template("result.html",order_id=order_ids,result=jsonify({'order_id':order_ids,'product_id':product_ids,'product_name':product_names}))
  # return flask.render_template("result.html",result=json.dumps({'order_id':list(order_ids),'products': list(submission.products.values)}))
  # return flask.render_template("result.html",result=jsonify({'order_ids':order_ids,'products':'#'.join(submission.products.values),'product_names':'#'.join(submission.product_name.values)}))





if __name__ == '__main__':
  user_all_info=pd.read_csv('user_all_info.csv')
  seluser_all_info=memory_decrease_by_column(user_all_info)
  order_dow_features=pd.read_csv('order_dow_features.csv')
  order_dow_features=memory_decrease_by_column(order_dow_features)
  order_hod_features=pd.read_csv('order_hod_features.csv')
  order_hod_features=memory_decrease_by_column(order_hod_features)
  order_dow_hod_features=pd.read_csv('order_dow_hod_features.csv')
  order_dow_hod_features=memory_decrease_by_column(order_dow_hod_features)
  ono_dsp_features=pd.read_csv('ono_dsp_features.csv')
  ono_dsp_features=memory_decrease_by_column(ono_dsp_features)
  products=pd.read_csv('products.csv',usecols=['product_id','product_name'])
  train_columns=pd.read_csv('train_columns.csv')
  train_columns=train_columns.train_columns.values
  X_train_statistics=pd.read_csv('X_train_statistics.csv')
  X_train_min=X_train_statistics.X_train_min
  X_train_min.index=train_columns
  X_train_max=X_train_statistics.X_train_max
  X_train_max.index=train_columns
  X_train_max_min=X_train_statistics.X_train_max_min
  X_train_max_min.index=train_columns
  app.run(host='0.0.0.0', port=7777)

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# In[3]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, median_absolute_error,mean_absolute_error


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# In[5]:


data = pd.read_csv('C:/Users/azade/Desktop/WCS/file.csv')
data.head(10)


# In[4]:


data.shape


# In[5]:


data[data.ITEM.duplicated(keep=False)].sort_values("DATE")


# In[6]:


data.describe()


# In[7]:


data.ITEM.unique()


# In[8]:


data.info()


# In[9]:


fig = plt.figure(figsize=(8, 5))
ax = sns.countplot(x=data['ITEM'], palette='Dark2')

plt.title('Items', size=14, color='green', pad=10)
plt.ylabel('Count')
plt.xlabel('')


# In[10]:


df= data.copy()


# In[11]:


# convert to date types
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')


# In[235]:


def date_features(df, label=None):
    df = df.copy()

    df['date'] = df.DATE
    df['month'] = df['date'].dt.strftime('%B')
    
    df['dayofweek'] = df['date'].dt.strftime('%A')
    df['quarter'] = df['date'].dt.quarter
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['dayofweek','quarter','month',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X
X, y = date_features(df, label='SALES')
df_new = pd.concat([X, y], axis=1)
df_new.head()


# In[236]:


fig, ax = plt.subplots(figsize=(14,5))

a = sns.barplot(x="month", y="SALES",data=df_new)
a.set_title("Store Sales Data in 2019",fontsize=15)


# In[237]:


fig,(ax1,ax2,ax3)= plt.subplots(nrows=3)
fig.set_size_inches(20,30)

monthAggregated = pd.DataFrame(df_new.groupby("month")["SALES"].sum()).reset_index().sort_values('SALES')
sns.barplot(data=monthAggregated,x="month",y="SALES",ax=ax1)
ax1.set(xlabel='Month', ylabel='Total Sales received')
ax1.set_title("Total Sales received By Month",fontsize=15)

monthAggregated = pd.DataFrame(df_new.groupby("dayofweek")["SALES"].sum()).reset_index().sort_values('SALES')
sns.barplot(data=monthAggregated,x="dayofweek",y="SALES",ax=ax2)
ax2.set(xlabel='dayofweek', ylabel='Total Sales received')
ax2.set_title("Total Sales received By Weekday",fontsize=15)

monthAggregated = pd.DataFrame(df_new.groupby("quarter")["SALES"].sum()).reset_index().sort_values('SALES')
sns.barplot(data=monthAggregated,x="quarter",y="SALES",ax=ax3)
ax3.set(xlabel='Quarter', ylabel='Total Sales received')
ax3.set_title("Total Sales received By Quarter",fontsize=15)


# In[7]:


#Read dataset
LILLE = pd.read_csv('C:/Users/azade/Desktop/WCS/LILLE.csv', sep=',',  encoding='latin-1')
LYON = pd.read_csv('C:/Users/azade/Desktop/WCS/LYON.csv', sep=',',  encoding='latin-1')
MARSEILLE = pd.read_csv('C:/Users/azade/Desktop/WCS/MARSEILLE.csv', sep=',',  encoding='latin-1')
BORDOEAUX = pd.read_csv('C:/Users/azade/Desktop/WCS/BORDEAUX.csv', sep=',',  encoding='latin-1')


# In[16]:


MARSEILLE.head()


# In[17]:


LILLE.shape


# In[18]:


MARSEILLE.info()


# In[19]:


LILLE.MAX_TEMPERATURE_C


# In[ ]:





# In[8]:


# convert to date types
LILLE['DATE'] = pd.to_datetime(LILLE['DATE'], errors='coerce')
LYON['DATE'] = pd.to_datetime(LYON['DATE'], errors='coerce')
MARSEILLE['DATE'] = pd.to_datetime(MARSEILLE['DATE'], errors='coerce')
BORDOEAUX['DATE'] = pd.to_datetime(BORDOEAUX['DATE'], errors='coerce')


# In[239]:


# make new columns
LILLE['month'] = LILLE['DATE'].dt.strftime('%B')
LILLE['dayofweek'] = LILLE['DATE'].dt.strftime('%A')
LILLE['dayofyear'] = LILLE['DATE'].dt.dayofyear
LILLE['quarter'] = LILLE['DATE'].dt.quarter
LILLE['dayofmonth'] = LILLE['DATE'].dt.day


# In[240]:


LYON['month'] = LYON['DATE'].dt.strftime('%B')
LYON['dayofyear'] = LYON['DATE'].dt.strftime('%d')
LYON['dayofweek'] = LYON['DATE'].dt.strftime('%A')
LYON['quarter'] = LYON['DATE'].dt.quarter

LYON['dayofmonth'] = LYON['DATE'].dt.day


# In[241]:


MARSEILLE['month'] = MARSEILLE['DATE'].dt.strftime('%B')

MARSEILLE['dayofweek'] = MARSEILLE['DATE'].dt.strftime('%A')
MARSEILLE['quarter'] = MARSEILLE['DATE'].dt.quarter
MARSEILLE['dayofyear'] = MARSEILLE['DATE'].dt.dayofyear
MARSEILLE['dayofmonth'] = MARSEILLE['DATE'].dt.day


# In[9]:


BORDOEAUX['month'] = BORDOEAUX['DATE'].dt.strftime('%B')

BORDOEAUX['dayofweek'] = BORDOEAUX['DATE'].dt.strftime('%A')
BORDOEAUX['quarter'] = BORDOEAUX['DATE'].dt.quarter
BORDOEAUX['dayofyear'] = BORDOEAUX['DATE'].dt.dayofyear
BORDOEAUX['dayofmonth'] = BORDOEAUX['DATE'].dt.day


# In[29]:


sns.set(font_scale=0.9)
plt.figure(figsize=(20,18))

plt.subplot(2,2,1)
ax = sns.boxplot(data=LILLE, x='month', y='MAX_TEMPERATURE_C', palette='Blues')
plt.title('Max of temp for LILLE')
plt.xlabel("")
plt.subplot(2,2,2)
sns.boxplot(data=LYON, x='month', y='MAX_TEMPERATURE_C', palette='Blues')
plt.title('Max of temp for LYON')
plt.xlabel("")
plt.subplot(2,2,3)
sns.boxplot(data=MARSEILLE, x='month', y='MAX_TEMPERATURE_C', palette='Blues')
plt.title('Max of temp for MARSEILLE')
plt.xlabel("")
plt.subplot(2,2,4)
sns.boxplot(data=BORDOEAUX, x='month', y='MAX_TEMPERATURE_C', palette='Blues')
plt.title('Max of temp for BORDOEAUX')
plt.xlabel("")


# In[ ]:





# In[31]:


sns.set(font_scale=0.9)
plt.figure(figsize=(20,18))

plt.subplot(2,2,1)
sns.boxplot(data=LILLE, x='month', y='MIN_TEMPERATURE_C', palette='pink')
plt.title('Min of temp for LILLE')
plt.xlabel("")
plt.subplot(2,2,2)
sns.boxplot(data=LYON, x='month', y='MIN_TEMPERATURE_C', palette='pink')
plt.title('Min of temp for LYON')
plt.xlabel("")
plt.subplot(2,2,3)
sns.boxplot(data=BORDOEAUX, x='month', y='MIN_TEMPERATURE_C', palette='pink')
plt.title('Min of temp for MARSEILLE')
plt.xlabel("")
plt.subplot(2,2,4)
sns.boxplot(data=BORDOEAUX, x='month', y='MIN_TEMPERATURE_C', palette='pink')
plt.title('Min of temp for BORDOEAUX')
plt.xlabel("")


# In[32]:


sns.set(font_scale=0.9)
plt.figure(figsize=(20,18))

plt.subplot(2,2,1)
sns.lineplot(data=LILLE, x='month', y='WINDSPEED_MAX_KMH', palette='Blues')
plt.title('WINDSPEED_MAX_KMH for LILLE')
plt.xlabel("")
plt.subplot(2,2,2)
sns.lineplot(data=LYON, x='month', y='WINDSPEED_MAX_KMH', palette='Blues')
plt.title('WINDSPEED_MAX_KMH for LYON')
plt.xlabel("")
plt.subplot(2,2,3)
sns.lineplot(data=MARSEILLE, x='month', y='WINDSPEED_MAX_KMH', palette='Blues')
plt.title('WINDSPEED_MAX_KMH for MARSEILLE')
plt.xlabel("")
plt.subplot(2,2,4)
sns.lineplot(data=BORDOEAUX, x='month', y='WINDSPEED_MAX_KMH', palette='Blues')
plt.title('WINDSPEED_MAX_KMH for BORDOEAUX')
plt.xlabel("")


# In[33]:


sns.set(font_scale=0.9)
plt.figure(figsize=(20,18))

plt.subplot(2,2,1)
ax = sns.countplot(x=LILLE['OPINION'], palette='Dark2')
total = len(LILLE['OPINION'])
for tick in ax.get_xticklabels():
    tick.set_rotation(30)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100 * p.get_height()/total), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.title('Percentage of weather Types for Lille', size=14, color='green', pad=10)
plt.ylabel('Count')
plt.xlabel("")
plt.subplot(2,2,2)

ax = sns.countplot(x=LYON['OPINION'], palette='Dark2')
total = len(LYON['OPINION'])
for tick in ax.get_xticklabels():
    tick.set_rotation(30)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100 * p.get_height()/total), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.title('Percentage of weather Types for LYON', size=14, color='green', pad=10)
plt.ylabel('Count')
plt.xlabel('')

plt.subplot(2,2,3)
ax = sns.countplot(x=MARSEILLE['OPINION'], palette='Dark2')
total = len(MARSEILLE['OPINION'])
for tick in ax.get_xticklabels():
    tick.set_rotation(30)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100 * p.get_height()/total), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.title('Percentage of weather Types for MARSEILLE', size=14, color='green', pad=10)
plt.ylabel('Count')
plt.xlabel("")
plt.subplot(2,2,4)
ax = sns.countplot(x=BORDOEAUX['OPINION'], palette='Dark2')
total = len(BORDOEAUX['OPINION'])
for tick in ax.get_xticklabels():
    tick.set_rotation(30)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100 * p.get_height()/total), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.title('Percentage of weather Types for BORDOEAUX', size=14, color='green', pad=10)
plt.ylabel('Count')
plt.xlabel("")


# In[ ]:





# In[12]:


# Merge sale dataset with bordeaux weather
df_merg = pd.merge(df, BORDOEAUX, how="left",on="DATE")


# In[13]:


df_merg.shape


# In[245]:


df_merg.info()


# In[246]:


# Heatmap for BORDOEAUX
sns.set(font_scale=0.8)
plt.figure(figsize=(20, 10))

df_corr = df_merg.corr()
sns.heatmap(df_corr, 
            xticklabels = df_corr.columns.values,
            yticklabels = df_corr.columns.values,
            annot = True);

plt.title("Corrolation for BORDOEAUX", fontsize =20)


# In[247]:


df_merg.columns


# In[248]:


# Drop high correllated variables
df_d= df_merg.drop(columns=['MIN_TEMPERATURE_C','TEMPERATURE_MORNING_C','TEMPERATURE_NOON_C',
                      'TEMPERATURE_EVENING_C','HEATINDEX_MAX_C',
                      'DEWPOINT_MAX_C','WINDTEMP_MAX_C'])


# In[249]:


# Heat map for LYON
sns.set(font_scale=0.8)
plt.figure(figsize=(20, 10))

df_corr = df_d.corr()
sns.heatmap(df_corr, 
            xticklabels = df_corr.columns.values,
            yticklabels = df_corr.columns.values,
            annot = True);

plt.title("Corrolation for BORDOEAUX", fontsize =20)


# In[251]:


df_d.info()


# In[253]:


df_d.select_dtypes(include=['object']).columns.tolist()


# In[254]:


df_d.select_dtypes(include=['int64']).columns.tolist()


# In[255]:


df_d.select_dtypes(include=['float64']).columns.tolist()


# ### ML

# In[311]:



# split dataset into train test sets
X= df_d.drop(columns=['SALES','DATE'], axis=1)
y = df_d['SALES']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=34)


# traitement des variables discrete 
numeric_dis_features = [ 'MAX_TEMPERATURE_C',
                         'WINDSPEED_MAX_KMH',
                         'HUMIDITY_MAX_PERCENT',
                         'PRESSURE_MAX_MB',
                         'WEATHER_CODE_MORNING',
                         'WEATHER_CODE_NOON',
                         'WEATHER_CODE_EVENING',
                         'TOTAL_SNOW_MM',
                         'UV_INDEX',
                         'quarter',
                         'dayofyear',
                         'dayofmonth']
numeric_dis_transformer = Pipeline(steps=[('scaler', StandardScaler())])




# traitement des variables continues :  standardcaler
numeric_con_features = ['PRECIP_TOTAL_DAY_MM',
                         'VISIBILITY_AVG_KM',
                         'CLOUDCOVER_AVG_PERCENT',
                         'SUNHOUR']
numeric_con_transformer = Pipeline(steps=[ ('scaler', StandardScaler())])



categorical_features = ['ITEM', 'OPINION', 'month','dayofweek']
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])


preprocessor = ColumnTransformer(transformers=[('num', numeric_dis_transformer, numeric_dis_features),
                                               ('num_cat', numeric_con_transformer, numeric_con_features),
                                               ('cat', categorical_transformer, categorical_features)])


from sklearn import set_config 
set_config(display='diagram')

preprocessor.fit(X_train, y_train)


# In[263]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# #### RandomForestRegressor

# In[331]:


# RandomForestRegressor
pipe_RFR = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', TransformedTargetRegressor(RandomForestRegressor()))])

param_RFR = {
             'model__regressor__max_features' : ['sqrt', 'log2'],
             'model__regressor__n_estimators':[100,250],
             'model__regressor__max_depth':[5,10,15],
             'model__regressor__min_samples_split': [2, 5, 10],
             'model__regressor__bootstrap' : [True, False],
             'model__regressor__min_samples_leaf': [1,2,5],
             'model__regressor__n_jobs': [-1]}

grid_RFR = GridSearchCV( pipe_RFR, param_RFR,cv=4, verbose=5)
grid_RFR.fit(X_train, y_train)


# In[270]:


grid_RFR.best_params_


# In[276]:


# RandomForestRegressor best params
RFR = Pipeline(steps=[ ('preprocessor', preprocessor),
                             ('model',TransformedTargetRegressor(regressor= RandomForestRegressor(bootstrap= False,
                  max_depth= 15, max_features= 'sqrt', min_samples_leaf= 2,min_samples_split= 10,
                  n_estimators = 100, n_jobs= -1  )))])
RFR_fit = RFR.fit(X_train, y_train)
RFR_Score= RFR_fit.score(X_test, y_test)


# In[277]:


RFR_Score


# In[281]:


print("score d'entrainement = ",RFR_Score,"\n")
y_pred = RFR_fit.predict(X_test)
MAE_RFR= mean_absolute_error(y_test,y_pred)
RMSE_RFR= mean_squared_error(y_test,y_pred , squared= False)
MSE_RFR = mean_squared_error(y_test,y_pred)

print("score de la prédiction:")#, accuracy_score(y_test, y_pred)), 
print("MAE = ",MAE_RFR)
print("RMSE = ",RMSE_RFR)
print("MSE = ",MSE_RFR)


# ## GradientBoostingRegressor

# In[343]:


# GradientBoostingRegressor
pipe_gbr = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', TransformedTargetRegressor(GradientBoostingRegressor()))])
   
grid_gbr = {
             'model__regressor__n_estimators':np.linspace(800,1200,3).astype(int),
             'model__regressor__learning_rate':[0.1,0.13, 0.16],
             'model__regressor__max_depth':[4, 6, 8],
             'model__regressor__subsample':[0.9, 1, 1.2]}

gs_gbr = GridSearchCV(estimator=pipe_gbr,
                      param_grid=grid_gbr,
                      cv=4,
                      verbose=5)

gs_gbr.fit(X_train, y_train)


# In[344]:


gs_gbr.best_params_


# In[476]:


# GradientBoostingRegressor best params
GBR= Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', GradientBoostingRegressor(n_estimators=1000, learning_rate=0.13,
                                      max_depth=6, subsample=1))])

GBR_fit= GBR.fit(X_train, y_train)
GBR_Score= GBR.score(X_test, y_test)


# In[346]:


print("score d'entrainement = ",GBR_Score,"\n")
y_pred = GBR.predict(X_test)
MAE_gbr= mean_absolute_error(y_test,y_pred)
RMSE_gbr = np.sqrt(mean_squared_error(y_test,y_pred))
MSE_gbr = mean_squared_error(y_test,y_pred)

print("score de la prédiction:")#, accuracy_score(y_test, y_pred)), 
print("MAE = ",MAE_gbr)
print("RMSE = ",RMSE_gbr)
print("MSE = ",MSE_gbr)


# ### Seprate Vector Regression

# In[385]:


from sklearn.svm import SVR
# Seprate Vector Regression
pipe_SVR = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', TransformedTargetRegressor(SVR())) ])
   
param_SVR = { 'model__regressor__C' : range(200,400,10),
              'model__regressor__epsilon' : [35, 40, 45],
              'model__regressor__kernel':['poly','rbf'],
              'model__regressor__gamma' : ['auto', 'scale'],
              'model__regressor__degree' : [1e-63, 1e-53,1e-43 ]
                }

GS_SVR = GridSearchCV(pipe_SVR,  param_SVR, cv=4,verbose=5)
                      
GS_SVR.fit(X_train, y_train)


# In[386]:


GS_SVR.best_params_


# In[355]:


GS_SVR.best_score_


# In[387]:


# SVR best params
SVR = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', TransformedTargetRegressor(SVR(C=200, degree=1e-63, epsilon=45, gamma= 'auto',
                                                                   kernel= 'rbf'))) ])
   
SVR_fit = SVR.fit(X_train, y_train)
SVR_Score = SVR.score(X_test, y_test)
SVR_Score


# In[403]:


print("score d'entrainement = ",SVR_Score,"\n")
y_pred = SVR.predict(X_test)
MAE_SVR= mean_absolute_error(y_test,y_pred)
RMSE_SVR = mean_squared_error(y_test,y_pred, squared= False)
MSE_SVR = mean_squared_error(y_test,y_pred)

print("score de la prédiction:")#, accuracy_score(y_test, y_pred)), 
print("MAE = ",MAE_SVR)
print("RMSE = ",RMSE_SVR)
print("MSE = ",MSE_SVR)


# In[ ]:





# In[ ]:





# In[395]:


# ExtraTreesRegressor
pipe_ETR = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', TransformedTargetRegressor(ExtraTreesRegressor())) ])
   
grid_ETR = [{
            'model__regressor__n_estimators' :range(1,100,10),
            'model__regressor__min_samples_split': [3, 4,5, 6],
            'model__regressor__min_samples_leaf': [ 3, 4, 5 6],
            'model__regressor__max_features': ['sqrt', 'log2'],
            'model__regressor__bootstrap': [False, True],
            'model__regressor__n_jobs': [None, -1]
                        }]

GS_ETR = GridSearchCV(pipe_ETR, grid_ETR, cv=4, verbose=5)

GS_ETR.fit(X_train, y_train)


# In[396]:


GS_ETR.best_params_


# In[398]:


#ExtraTreesRegressor best params
ETR= Pipeline(steps=[('preprocessor', preprocessor),
                              ('model',ExtraTreesRegressor(n_estimators= 11 ,min_samples_leaf= 5,
                                min_samples_split= 5, bootstrap= False , max_features = 'sqrt', n_jobs= None))])
ETR_fit= ETR.fit(X_train, y_train)
ETR_score= ETR.score(X_test, y_test)
ETR_score


# In[402]:


print("score d'entrainement = ",ETR_score,"\n")
y_pred = ETR.predict(X_test)
MAE_ETR= mean_absolute_error(y_test,y_pred)
RMSE_ETR = np.sqrt(mean_squared_error(y_test,y_pred))
MSE_ETR = mean_squared_error(y_test,y_pred)

print("score de la prédiction:")
print("MAE = ",MAE_ETR)
print("RMSE = ",RMSE_ETR)
print("MSE = ",MSE_ETR)


# In[ ]:





# In[404]:


model_df = {'models': ['Gradient Boosting', 'RandomForestRegressor', 'SVR', 'ExtraTreesRegressor'],
            'mean_absolute_error': [MAE_gbr, MAE_RFR, MAE_SVR,MAE_ETR ],
            'Root_mean_squared_error': [RMSE_gbr, RMSE_RFR, RMSE_SVR, RMSE_ETR],
            
            'Score': [GBR_Score,RFR_Score, SVR_Score, ETR_score]}
result= pd.DataFrame(model_df)
result


# In[405]:


sns.set(font_scale=1.1)
plt.figure(figsize=(20,8))
plt.suptitle("MAE & RMSE sur les donnees ",fontsize=22)
plt.subplot(1,2,1)
sns.barplot(x= result['models'] ,y=result['mean_absolute_error'])
plt.title(" mean error", fontsize=17)

plt.subplot(1,2,2)
sns.barplot(x= result['models'] ,y=result['Root_mean_squared_error'])
plt.title(" variance error", fontsize=17)


# In[406]:


feature_importances= GBR.named_steps['model'].feature_importances_
feature_importances


# In[407]:


ohe_feature_nemes= GBR['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names(categorical_features)


# In[408]:


ohe_feature_nemes


# In[409]:


numeric_features_names= np.concatenate([numeric_dis_features,numeric_con_features])
numeric_features_names


# In[413]:


feature_importances_names= np.concatenate([numeric_features_names,ohe_feature_nemes,])
feature_importances_names.shape


# In[414]:


#zip coeff & names together and make  DataFrame
zipped= zip(feature_importances_names,feature_importances)
fi_df= pd.DataFrame(zipped, columns=['feature','importances'])
fi_df


# In[416]:


sns.set(font_scale=1.2)
plt.figure(figsize=(10,8))
sns.barplot(y='feature', x='importances', data= fi_df.sort_values(by= 'importances', ascending = False).head(20))


# In[ ]:





# In[16]:


pr = pd.read_csv('C:/Users/azade/Desktop/WCS/forcast.csv')
pr.head(10)


# In[531]:


pr.info()


# In[14]:


df_merg.columns


# In[17]:


pr.columns


# In[23]:


df_fin = df_merg.filter(items=['MAX_TEMPERATURE_C', 'MIN_TEMPERATURE_C', 'WINDSPEED_MAX_KMH',
       'PRECIP_TOTAL_DAY_MM', 'HUMIDITY_MAX_PERCENT', 'VISIBILITY_AVG_KM',
       'PRESSURE_MAX_MB', 'CLOUDCOVER_AVG_PERCENT','SALES'])


# In[24]:


df_fin.info()


# In[26]:


X_f= df_fin.drop(columns=['SALES'], axis=1)
y_f = df_fin['SALES']

X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(X_f,y_f, test_size=0.2, random_state=34)


# In[36]:


scale= StandardScaler()
X_tr_std= scale.fit_transform(X_f_train)
X_ts_std= scale.fit_transform(X_f_test)


# In[97]:


#XGBoost hyper-parameter tuning
def hyperParameterTuning(X_f_train, y_f_train):
    param_tuning = {
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.5, 0.7],
                    'n_estimators' : [100, 200, 500],
                    
    }

    xgb_model = GradientBoostingRegressor() 

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                         
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)
    gsearch.fit(X_f_train,y_f_train)

    return gsearch.best_params_


# In[98]:


hyperParameterTuning(X_f_train, y_f_train)


# In[99]:


# GradientBoostingRegressor best params
GBR= GradientBoostingRegressor(n_estimators=100, learning_rate=0.01,
                                      max_depth=3, subsample=0.5)

GBR_fit_f= GBR.fit(X_tr_std, y_f_train)
GBR_score_m= GBR.score(X_ts_std, y_f_test)


# In[100]:


print("score d'entrainement = ",GBR_score_m,"\n")
y_pred = GBR.predict(X_ts_std)
MAE_GBR_f= mean_absolute_error(y_f_test,y_pred)
RMSE_GBR_f = np.sqrt(mean_squared_error(y_f_test,y_pred))


print("score de la prédiction:")
print("MAE = ",MAE_GBR_f)
print("RMSE = ",RMSE_GBR_f)


# In[ ]:





# In[ ]:





# In[ ]:





# In[59]:


pip install hyperopt 


# In[65]:


pip install xgboost


# In[66]:


# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import xgboost as xgb


# In[61]:


space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }


# In[74]:


def objective(space):
    clf=xgb.XGBClassifier(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( X_f_train, y_f_train), (  X_f_test, y_f_test)]
    
    clf.fit(X_f_train, y_f_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)
    

    pred = clf.predict(X_f_test)
    accuracy = accuracy_score(y_f_test, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }


# In[75]:


trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)


# In[ ]:





# In[ ]:





# In[ ]:





# In[525]:


# pr.DATE = pr.DATE.str.replace('-', '').astype(int)


# In[ ]:





# In[ ]:





import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import GaussianNB

st.markdown('<style>body{background-color: #E8E8E8;}</style>',unsafe_allow_html=True)


st.title("Online Vehicle Identification: Machine Learning Model")

@st.cache
def load_data(classifier_name):
   data = pd.read_csv('Traffic_Violations_ds.csv', low_memory=False)
   del data['Date Of Stop']
   del data['Time Of Stop']
   data.dropna(axis=0, subset=['Latitude'], inplace=True)
   data.dropna(axis=0, subset=['Longitude'], inplace=True)
   data.dropna(axis=0, subset=['Year'], inplace=True)
   data.dropna(axis=0, subset=['Article'], inplace=True)
   data.dropna(axis=0, subset=['Geolocation'], inplace=True)
   data['Description'].fillna('Other', inplace = True)
   data['Location'].fillna('Other', inplace = True)
   data['State'].fillna('Other', inplace = True)
   data['Make'].fillna('Other', inplace = True)
   data['Model'].fillna('Other', inplace = True)
   data['Color'].fillna('Other', inplace = True)
   data['Driver City'].fillna('Other', inplace = True)
   data['Driver State'].fillna('Other', inplace = True)
   data['DL State'].fillna('Other', inplace = True)
   return data



classifier_name = st.selectbox("Select Classifier",("Naive Bayes","K-Nearest Neighbors","Random Forest Classifier"))
class_variable = st.selectbox("Class Variables",("Contributed To Accident","Belts"))
df = load_data(classifier_name)
df_fs = df.copy()
df_fs = df_fs.head(10000)
dataset = df_fs.copy()

st.sidebar.header("Predict Class Variable Using Random Forest Classifier")
st.sidebar.subheader("Traffic Violation:")

varx = st.sidebar.selectbox("Select Veriable To Predict:",("Gender","Race","Contributed To Accident","Alcohol"))
year = st.sidebar.selectbox("Select Year:",(2001,2006,2010))
lat = st.sidebar.slider('Latitude', min_value=-94.610988, max_value=40.111822)
lon = st.sidebar.slider('Longitude', min_value=-77.732495, max_value=41.543160)

dataset = dataset[['Latitude', 'Longitude','Year', varx]]
df_X = dataset.drop(varx, axis=1) 
y = dataset[varx]

X = pd.get_dummies(df_X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

rf = RandomForestClassifier(random_state=10)
rf.fit(X_train, y_train)
Race = rf.predict([[lat,lon,year]])
st.sidebar.write('Year Of Incident : ',year)
st.sidebar.write('Latitude : ',lat)
st.sidebar.write('Longitude : ',lon)
st.sidebar.write(varx,' is : '+Race[0])

fig = plt.figure()
b=sns.countplot(x=varx, data = dataset)
plt.title('Frequency of '+varx)
st.sidebar.pyplot(fig)
#---------------------------------------------------------------------------------------------------------

@st.cache
def load_data2(parking_data):
   data = pd.read_csv('Parking_Violations_Issued.csv', low_memory=False)
   del data['Summons Number']
   data['Violation In Front Of Or Opposite'].fillna('F', inplace = True)
   data.fillna('other', inplace = True)
   return data

st.sidebar.subheader("Parking Violation Issued:")
var_y = st.sidebar.selectbox("Select Variable To Predict:",("Plate Type","Plate ID","Vehicle Make","Vehicle Body Type"))
vc = st.sidebar.slider('Violation Code', min_value=0, max_value=99)
Street1 = st.sidebar.slider('Street Code 1', min_value=0, max_value=98020)
Street2 = st.sidebar.slider('Street Code 2', min_value=0, max_value=98260)
Street3 = st.sidebar.slider('Street Code 3', min_value=0, max_value=98260)

data_parking = load_data2(var_y)
df_new = data_parking.copy()
df_ds = data_parking.copy()
df_ds = df_ds[['Street Code1','Street Code2','Street Code3','Violation Code',var_y]]

df_X = df_ds.drop(var_y, axis=1) 
y = df_ds[var_y]
X = pd.get_dummies(df_X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

pred_val= knn.predict([[Street1,Street2,Street3,vc]])
st.sidebar.write('Violation Code : ',vc)
st.sidebar.write('Street 1 : ',Street1)
st.sidebar.write('Street 2 : ',Street2)
st.sidebar.write('Street 3 : ',Street3)
st.sidebar.write(var_y,' is : '+pred_val[0])

fig = plt.figure()
sns.distplot(df_ds["Violation Code"], bins=10)
st.sidebar.pyplot(fig)
#---------------------------------------------------------------------------------------------------------

st.write(df_fs.head(5))
st.write("Shape of dataset:", df_fs.shape)

#Encoding the Raw Dataset 
categorical = ['Agency','SubAgency','Description','Location','Accident','Belts','Personal Injury','Property Damage','Fatal','Commercial License','HAZMAT','Commercial Vehicle','Alcohol','Work Zone','State',
              'VehicleType','Make', 'Model','Color','Violation Type','Charge','Article','Contributed To Accident','Race','Gender','Driver City','Driver State','DL State','Arrest Type','Geolocation']

d = defaultdict(LabelEncoder)
df_fs[categorical] = df_fs[categorical].apply(lambda x: d[x.name].fit_transform(x.astype(str)))



if(class_variable == "Contributed To Accident"):
   X = df_fs.drop('Contributed To Accident', axis=1) 
   y = df_fs['Contributed To Accident']
   X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=7)
   
elif (class_variable == "Belts"):
   X = df_fs.drop('Belts', axis=1) 
   y = df_fs['Belts']
   X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=7)
else:
   #default Race
   X = df_fs.drop('Belts', axis=1) 
   y = df_fs['Belts']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=10)


if(classifier_name == "K-Nearest Neighbors"):
   st.success(classifier_name)
   
   #Classification-KNN
   knn = KNeighborsClassifier(n_neighbors=5)
   knn.fit(X_train, y_train)

   y_pred = knn.predict(X_test)

   confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)

   st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
   st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, pos_label=0)))
   st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred, pos_label=0)))
   st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

   st.title("Class Variable: " + class_variable)

   col1, col2 = st.beta_columns([20,20])
   #plot knn
   with col1:
      prob_KNN = knn.predict_proba(X_test)
      prob_KNN = prob_KNN[:, 1]
      fpr_knn, tpr_knn, thresholds_DT = roc_curve(y_test, prob_KNN)

      fig = plt.figure()
      plt.plot(fpr_knn, tpr_knn, color='red', label='KNN') 
      plt.plot([0, 1], [0, 1], color='green', linestyle='--')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('K-Nearest Neighbors (KNN) Curve')
      plt.legend()
      st.pyplot(fig)
   
   with col2:
      prec_knn, rec_knn, thresholds_DT = precision_recall_curve(y_test, prob_KNN)
      fig = plt.figure()
      plt.plot(prec_knn, rec_knn, color='red', label='KNN') 
      plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.title('Precision-Recall Curve')
      plt.legend()
      st.pyplot(fig)
   
elif(classifier_name == "Naive Bayes"):

   st.success(classifier_name)

   col1, col2 = st.beta_columns([20,20])

    #Classification-NB
   nb = GaussianNB()
   nb.fit(X_train, y_train)
   y_pred = nb.predict(X_test)
   confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)
   st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
   st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, pos_label=0)))
   st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred, pos_label=0)))
   st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))
   
   with col1:
      prob_NB = nb.predict_proba(X_test)
      prob_NB = prob_NB[:, 1]
      fpr_NB, tpr_NB, thresholds_DT = roc_curve(y_test, prob_NB) 

      fig = plt.figure()
      plt.plot(fpr_NB, tpr_NB, color='orange', label='NB')
      plt.plot([0, 1], [0, 1], color='green', linestyle='--')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver Operating Characteristic (ROC) Curve')
      plt.legend()
      st.pyplot(fig)

   with col2:
      prec_NB, rec_NB, threshold_NB = precision_recall_curve(y_test, prob_NB)
      fig = plt.figure()
      plt.plot(prec_NB, rec_NB, color='orange', label='NB') 
      plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.title('Precision-Recall Curve')
      plt.legend()
      st.pyplot(fig)

else:
   st.success(classifier_name)
   col1, col2 = st.beta_columns([20,20])

   rf = RandomForestClassifier(random_state=10)
   rf.fit(X_train, y_train)
   y_pred = rf.predict(X_test)
   confusion_majority = confusion_matrix(y_test, y_pred)

   confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)

   st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
   st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, pos_label=0)))
   st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred, pos_label=0)))
   st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

   #plot RF
   with col1:
      fig = plt.figure()
      prob_RF = rf.predict_proba(X_test)
      prob_RF = prob_RF[:, 1]
      fpr_rf, tpr_rf, thresholds_DT = roc_curve(y_test, prob_RF)
      
      plt.plot(fpr_rf, tpr_rf, color='purple', label='RF') 
      plt.plot([0, 1], [0, 1], color='green', linestyle='--')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver Operating Characteristic (ROC) Curve')
      plt.legend()
      st.pyplot(fig)
      
   with col2:
      prec_RF, rec_RF, thresholds_DT = precision_recall_curve(y_test, prob_RF)
      fig = plt.figure()
      plt.plot(prec_RF, rec_RF, color='purple', label='RF') 
      plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.title('Precision-Recall Curve')
      plt.legend()
      st.pyplot(fig)

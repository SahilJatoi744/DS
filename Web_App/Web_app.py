# import libraries
from pkgutil import get_data
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# app ki heading
st.write("""
# Explore different ML models and datasets
Daikhtay han kon sa best ha in may say? 
""")

# data set k name ak box may daal k sidebar pay laga do
dataset_name = st.sidebar.selectbox('Select Dataset',('Iris','Breast Cancer','Wine'))

# or isi k nichy classifier k nam ak debay may dal do
classifier_name = st.sidebar.selectbox('Select Classifier',('KNN','SVM','Random Forest'))

# ab hum nay ak function define krna hai dataset ko load krnay ke liye
def get_dataset(dataset_name):
    data = None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x= data.data
    y = data.target
    return x,y

# ab is function ko bula lay gayn or x,y variable k equal rakh layn gay
X,y = get_dataset(dataset_name)

# ab hum apnay data set ki shape ko print kr dayn gay
st.write("Shape of dataset:",X.shape)
st.write("Number of classes",len(np.unique(y)))

# next hum different classifier k parameter ko user input may add karayn gay
def add_parameter_ui(classifier_name):
    params = dict() # create an empty dictionary
    if classifier_name == 'SVM':
        C=  st.sidebar.slider('C',0.01,10.0)
        params['C'] = C # its the degree of correct classifier
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K  # its the number of nearest neighbour
    else:
        max_depth = st.sidebar.slider('max_depth',2,15)
        params['max_depth'] = max_depth # depth of every tree that grow in random forest
        n_estimators = st.sidebar.slider('n_estimators',1,100)
        params['n_estimators'] = n_estimators # number of trees
    return params

# ab is function ko bula lay gayn or params variable k equal rakh layn gay
params = add_parameter_ui(classifier_name)

# a hum classifier bnayen gay based on classifier_name and params
def get_classifier(classifier_name,params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(random_state=1234)
    return clf

# check box to show code
if st.checkbox("Show code"):
    with st.echo():
        # ab is function ko bula lay gayn or clf varaible k equal rakh layn gay
        clf = get_classifier(classifier_name, params)

        # ab hum apnay dataset ko test and train data may split kr laytay han by 80/20 ratio
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234)

        # ab hum nay apnay classifier ki training karni hy
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)


        # model ka accuracy score check layna ha or isay app pay print kr dayna ha
        acc = accuracy_score(y_test, y_pred)



# ab is function ko bula lay gayn or clf varaible k equal rakh layn gay
clf = get_classifier(classifier_name,params)

# ab hum apnay dataset ko test and train data may split kr laytay han by 80/20 ratio
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

# ab hum nay apnay classifier ki training karni hy
clf.fit(X_train,y_train)
y_pred= clf.predict(X_test)




# model ka accuracy score check layna ha or isay app pay print kr dayna ha
acc = accuracy_score(y_test,y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = {acc}')

#### PLOT DATSET ####
# ab hum apnay sary sary features ka 2 dimentional plot pay draw kr dayn gay using pca
pca = PCA(2)
X_projected = pca.fit_transform(X)

# ab hum apna data 0 or 1 dimension may slice kr dayn gay
x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

# plt.show()
st.pyplot(fig)

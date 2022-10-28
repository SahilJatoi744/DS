import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from PIL import Image

# Title
st.title("Diabetes Prediction App")

#Lodainng the dataset

d_data=pd.read_csv("diabetes.csv")



options=st.sidebar.radio("Main Menu",["Introduction","Exploratory Data Analysis","Data Visualization","Scaling the Data","Model Building"])



def Content(options):

    if options == "Introduction":
        return intro()
    elif options == "Exploratory Data Analysis":
        return EDA()
    elif options == "Data Visualization":
        return visualizations()
    elif options == "Scaling the Data":
        return Scaling_Data()
    elif options == "Model Building":
        return Model_Building()

d_data_copy = d_data.copy(deep = True)

def intro():
    st.header("Introduction to the Dataset")
    st.write("In this Web App, we will be predicting that whether the patient has diabetes or not on the basis of the features we will provide to our machine learning model, and for that, we will be using the famous Pima Indians Diabetes Database")
    st.write("Data analysis: Here one will get to know about how the data analysis part is done in a data science life cycle.")
    st.write("Exploratory data analysis: EDA is one of the most important steps in the data science project life cycle and here one will need to know that how to make inferences from the visualizations and data analysis")
    st.write("Model building: Here we will be using 4 ML models and then we will choose the best performing model.")
    st.write("Saving model: Saving the best model using pickle to make the prediction from real data")
    image = Image.open('Image_1.jpg')
    st.image(image, caption='Diabetes Prediction', use_column_width=True)



        
def EDA():
    st.header("Exploratory Data Analysis")
    if st.checkbox("Show The Dataset"):
        st.write("### Enter the number of rows to view")
        rows = st.number_input("", min_value=0,value=5)
        if rows > 0:
            st.table(d_data.head(rows))
    if st.checkbox("Number of rows and columns"):
        st.write(f'Rows: {d_data.shape[0]}')
        st.write(f'Columns: {d_data.shape[1]}')
    if st.checkbox("Columns of the Dataset"):
        st.table(d_data.columns)
    if st.checkbox("Summary Statistics"):
        st.write(d_data.describe().T)
    if st.checkbox("Unique Values"):
        st.table(d_data.nunique())
    if st.checkbox("Indexing Values"):
        st.text(d_data.index)
    if st.checkbox("Missing Values in the Dataset"):
        st.table(d_data.isnull().sum())
        
    st.write("In this particular dataset all the missing values were given the 0 as a value which is not good for the authenticity of the dataset. Hence we will first replace the 0 value with the NAN value then start the imputation process")
    if st.checkbox("Replacing 0 with NAN"):
        d_data_copy[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]]=d_data_copy[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]].replace(0,np.NAN)
        st.write("The 0 values have been replaced with the NAN values")
        if st.checkbox("Missing Values in the Dataset After Replacing 0 with NAN"):
            st.table(d_data_copy.isnull().sum())

    st.write("Now we will start the imputation process and replace the NAN values with the mean of the respective columns so the following is the dataset after imputation process")
    if st.checkbox("Imputing the Missing Values"):
        d_data_copy['Glucose'].fillna(d_data_copy['Glucose'].mean(), inplace = True)
        d_data_copy['BloodPressure'].fillna(d_data_copy['BloodPressure'].mean(), inplace = True)
        d_data_copy['SkinThickness'].fillna(d_data_copy['SkinThickness'].median(), inplace = True)
        d_data_copy['Insulin'].fillna(d_data_copy['Insulin'].median(), inplace = True)
        d_data_copy['BMI'].fillna(d_data_copy['BMI'].median(), inplace = True)
        st.write("The missing values have been imputed")
        if st.checkbox("Dataset After Imputation"):
            st.write("### Enter the number of rows to view")
            row = st.number_input("",min_value=0,max_value=100,value=5)
            if row > 0:
                st.table(d_data_copy.head(row))
  
        if st.checkbox("Unique Values After Imputation Process"):
            st.table(d_data_copy.nunique())

def histograms():
    st.subheader("Distribution Plots Before Removing Null Values")
    st.write("The following are the distribution plots of the dataset before removing the null values")
    fig1=px.histogram(d_data,x="Pregnancies",y="Pregnancies",color="Outcome",title="Distribution of Pregnancies")
    st.plotly_chart(fig1)
    fig2=px.histogram(d_data,x="Glucose",y="Glucose",color="Outcome",title="Distribution of Glucose")
    st.plotly_chart(fig2)
    fig3=px.histogram(d_data,x="BloodPressure",y="BloodPressure",color="Outcome",title="Distribution of BloodPressure")
    st.plotly_chart(fig3)
    fig4=px.histogram(d_data,x="SkinThickness",y="SkinThickness",color="Outcome",title="Distribution of SkinThickness")
    st.plotly_chart(fig4)
    fig5=px.histogram(d_data,x="Insulin",y="Insulin",color="Outcome",title="Distribution of Insulin")
    st.plotly_chart(fig5)
    fig6=px.histogram(d_data,x="BMI",y="BMI",color="Outcome",title="Distribution of BMI")
    st.plotly_chart(fig6)
    fig7=px.histogram(d_data,x="DiabetesPedigreeFunction",y="DiabetesPedigreeFunction",color="Outcome",title="Distribution of DiabetesPedigreeFunction")
    st.plotly_chart(fig7)
    fig8=px.histogram(d_data,x="Age",y="Age",color="Outcome",title="Distribution of Age")
    st.plotly_chart(fig8)
    st.subheader("Interpretation of the Distribution Plots")
    st.write("The above distribution plots are of the dataset before imputing the missing values. The following are the inferences we can make from the above distribution plots")

def box_plots():
    st.subheader("Box Plots Before Removing Null Values")
    st.write("The following are the box plots of the dataset before removing the null values")
    fig1=px.box(d_data,y="Pregnancies",title="Box plot of Pregnancies")
    st.plotly_chart(fig1)
    fig2=px.box(d_data,y="Glucose",title="Box plot of Glucose")
    st.plotly_chart(fig2)
    fig3=px.box(d_data,y="BloodPressure",title="Box Plot of BloodPressure")
    st.plotly_chart(fig3)
    fig4=px.box(d_data,y="SkinThickness",title="Box Plot of SkinThickness")
    st.plotly_chart(fig4)
    fig5=px.box(d_data,y="Insulin",title="Box Plot of Insulin")
    st.plotly_chart(fig5)
    if st.checkbox("Interpretation of Box Plots"):
        st.write("The box plots of all columns shows some outliers but the Insulin column shows so many outliers in the dataset which is not good for the authenticity of the dataset and hence we will remove the outliers")
        if st.checkbox("Remove the Outliers in Insulin Column"):
            st.write("The outliers in the Insulin column have been removed")
            d_data_copy = d_data.copy(deep = True)
            d_data_copy = d_data_copy[d_data_copy['Insulin']<300]
            fig6=px.box(d_data_copy,y="Insulin",title="Box Plot of Insulin")
            st.plotly_chart(fig6)
def scatter_plot():
    plot_scatter = px.scatter(d_data,x="Age",y="Glucose",color="Outcome",title="Scatter Plot of Age vs Glucose")
    st.plotly_chart(plot_scatter)

def countplots():
    co_plot=px.bar(d_data['Outcome'].value_counts(),title="Count Plot of Outcome")
    st.plotly_chart(co_plot)
    st.write("The above count plot shows that the dataset is imbalanced as the number of people who have diabetes is less than the number of people who don't have diabetes")

def piechart():
    pie=px.pie(d_data,names="Outcome",title="Pie Chart of Outcome")
    st.plotly_chart(pie)
    st.write("The above pie chart shows that the dataset is imbalanced as the number of people who have diabetes is less than the number of people who don't have diabetes")
def correlation():
    st.write("The following is the correlation matrix of the dataset")
    corr=d_data.corr()
    h_map=sns.heatmap(corr,annot=True)
    st.pyplot(h_map.figure)
    st.write("The above correlation matrix shows that the columns Glucose and Outcome are highly correlated and hence we will use this column for our model")

def visualizations():

    st.header("Data Visualization")
    if st.checkbox("Histograms"):
        histograms()
    if st.checkbox("Boxplots"):
        box_plots()
    if st.checkbox("Countplots"):
        countplots()
    if st.checkbox("Pie Chart"):
        piechart()
    if st.checkbox("Scatter Plot"):
        scatter_plot()
    if st.checkbox("Heat Map"):
        correlation()

def Scaling_Data():
    sc_X = StandardScaler()
    X =  pd.DataFrame(sc_X.fit_transform(d_data_copy.drop(["Outcome"],axis = 1),), columns=['Pregnancies', 
    'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    if st.checkbox("Scaled Data"):
        st.write("### Enter the number of rows to view")
        rows = st.number_input("", min_value=0,value=5)
        if rows > 0:
            st.table(X.head(rows))
            st.write("That’s how our dataset will be looking like when it is scaled down or we can see every value now is on the same scale which will help our ML model to give a better result.")

def Model_Building():
    if st.checkbox("Input Data and Output Data"):
        X = d_data.iloc[:,0:8]
        y = d_data.iloc[:,8]
        if st.checkbox("X and Y Data"):
            st.write("### Enter the number of rows to view")
            rows = st.number_input("", min_value=0,value=5)
            if rows > 0:
                st.table(X.head(rows))
                st.table(y.head(rows))
        if st.checkbox("Splitting the Data"):
            train_size=st.selectbox("Train Size",[0.7,0.8,0.9])
            random_state=st.selectbox("Random State",[0,1,21,33,42])
            X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=train_size,random_state=random_state)
            st.write("The data has been splitted into train and test data")
            st.write("The shape of X_train is",X_train.shape)
            st.write("The shape of X_test is",X_test.shape)
            st.write("The shape of y_train is",y_train.shape)
            st.write("The shape of y_test is",y_test.shape)
        if st.checkbox("Model Selection"):
            classifier_name = st.selectbox("Select Classifier",("SVM","KNN","Random Forest","Desicion Tree"))
            def add_parameter(classifier_name):
                params=dict()
                if classifier_name=="SVM":
                    C=st.slider("C",0.01,10.0)
                    params["C"]=C
                elif classifier_name=="KNN":
                    K=st.slider("K",1,15)
                    params["K"]=K
                elif classifier_name=="Random Forest":
                    max_depth=st.slider("Max Depth",2,15)
                    params['max_depth']=max_depth
                    n_estimators=st.slider("Number of Estimators",1,200)
                    params['n_estimators']=n_estimators
                elif classifier_name=="Desicion Tree":
                    max_depth=st.slider("Max Depth",2,15)
                    params['max_depth']=max_depth
                    criterion=st.selectbox("Criterion",["gini","entropy"])
                    params['criterion']=criterion
                return params
            params=add_parameter(classifier_name)
            def get_classifier(classifier_name,params):
                clf=None
                if classifier_name=="SVM":
                    clf=SVC(C=params["C"])
                elif classifier_name=="KNN":
                    clf=KNeighborsClassifier(n_neighbors=params["K"])
                elif classifier_name=="Random Forest":
                
                    clf=RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],random_state=random_state)
                elif classifier_name=="Desicion Tree":
                    clf=DecisionTreeClassifier(criterion=params['criterion'],max_depth=params['max_depth'])
                return clf
            clf=get_classifier(classifier_name,params)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            st.write("The accuracy of the model is",accuracy_score(y_test,y_pred))
            st.write("The confusion matrix of the model is")
            st.table(confusion_matrix(y_test,y_pred))


Content(options)





    



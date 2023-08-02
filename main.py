from sklearn.model_selection import train_test_split
from turtle import title
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import statistics
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
#from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings('ignore')
import sys

#print(sys.argv)

path=os.getcwd()
#print(path)
import sys
#data=pd.DataFrame()
#with open(sys.argv[1], 'r') as f:
#    data=pd.read_csv(f.read())
#print (contents)
#data = pd.read_csv(contents)
#data.info()
CMV = os.getcwd()
string="\\"
cmd_arg=sys.argv
data = pd.read_csv(CMV+string+cmd_arg[1])
#data.info()
#data.info to check what all attributes we have in the dataset.
# To count how many columns we have in the dataset.
#data.describe()

#data.describe to give preliminary analysis of all the attributes.
#This gives count, mean, std, min , max etc.

#####Preprocessing#####
data.isnull().any()
#This is to check whall all columns have atleast one null value.
data.isnull().sum()
# To count how many null values we have in each of the column


data['fixed acidity'].fillna(data['fixed acidity'].mean(),inplace=True)
data['volatile acidity'].fillna(data['volatile acidity'].mean(),inplace=True)
data['citric acid'].fillna(data['citric acid'].mean(),inplace=True)
data['residual sugar'].fillna(data['residual sugar'].mean(),inplace=True)
data['chlorides'].fillna(data['chlorides'].mean(),inplace=True)
data['pH'].fillna(data['pH'].mean(),inplace=True)
data['sulphates'].fillna(data['sulphates'].mean(),inplace=True)
#Afer counting the number of zeroes we have to replace them with number.
#we can choose different methods to replace that empty null value.
#We chose to replace the null values with mean value of the non - null values.

data.head(10)
#The top 10 values of the processed subset
abc = data['quality']
from collections import Counter
#print(Counter(abc))
#We are using counter to see what are the range of values in quality column

#####Analysis#####
#Using Five metrics and using columns
tempFrame=data.drop('type', axis=1)
attr_list= tempFrame.columns.tolist() # get attribute names
#print(attr_list)
analysis={}
for ele in attr_list:
   analysis[ele]= [[tempFrame[ele].max(),tempFrame[ele].min()],tempFrame[ele].mean(),statistics.multimode(tempFrame[ele].tolist()),tempFrame[ele].std(),tempFrame[ele].median()]

print(pd.DataFrame(analysis,index=["Range","Mean","Mode","Standard Deviation","Median"]))

tempData=pd.DataFrame()
tempData=data
#tempData.drop('type', axis=1)
corrMatrix = tempData.corr()
#print (corrMatrix)
corrMatrix = data.corr()
#print (corrMatrix)
##Here we are finding out the correlation values between attributes of our dataset.

quals = [
    (data['quality'] >= 7),
     (data['quality'] <= 4)
]
rating = ['good', 'bad']
data['rating'] = np.select(quals, rating, default='Average')
data.rating.value_counts()
# We are dividing different quality of wines into three categories.
# Good, average, poor. This is to use later for visualization in pie chart.





data['type'].unique()
#####Visulaiztion#####

# ##Technique 1 - BarGraph
# sns.countplot(x=data['quality'])
# plt.title("Bar graph showing the distribution for Count of Quality Attribute")
# plt.show()
# To see the distribution of quality in the dataset.
# Observation 1 - We can see that major chunk of wines are in the quality range of 5,6,7

##technique2 - Pie Chart
total= data['quality'].count()


good = data.loc[(data['quality'] >=7 )]
average=data.loc[(data['quality'] >= 5) & (data['quality'] <= 6)]
poor = data.loc[(data['quality']<=5)]
goo = good.shape[0]
avera = average.shape[0]
poo = poor.shape[0]
#good_percent = good*100/total
print(goo, avera, poo)


labels = 'Quality = Good', 'Quality = Average', 'Quality = Poor'
sizes = [goo, avera, poo]
colors = ['Red', 'Blue','Green' ]
#Visualization Technique - 2
# # Plot pie chart
# plt.pie(sizes,  labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=90, )
# plt.title("Pie Chart visualizing quality in three categories ")
# plt.axis('equal')
# plt.show()
#
# #We can catogerized data and found out that major chunk of the data set is in average quality of wine
#
# plt.scatter(data["pH"],data["chlorides"])
# plt.title("Scatter Plot of Data points between pH and Chlorides")
# plt.show()
# #Here we we using a scatter plot. This is to visually see the data points between ph and cholirdes
#
# ##Visualization tecchnique - 3
# plt.figure(figsize=[14,6])
# sns.heatmap(tempData.corr(),annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
# plt.title("Correlation Matrix")
# plt.show()
# #one metric for correlation
# #To see how different attributes are related to each other
# # We can see that  'alcohol, sulphates, citric_acid & fixed_acidity' have psoitive corelation with 'quality'.
# # Observations -
# # Wine quality has positive correlation with quality.
# # Density has negative correlation with quality
#
# #Visualization Technique - 4
# #Scatter plot with regression line
# sns.regplot(x='quality', y='residual sugar', data=data,ci=None)
# plt.title("Scatter plot with regression line")
# plt.show()
# #Observation - Residual Sugar values are going down for Excellent quallity wines whereas for poor quality wines residual sugar is going up.
#
# #Visualization Technique - 5
# plot1 = sns.boxplot(x="quality", y='alcohol', data = data)
# plot1.set(xlabel='Wine Quality', ylabel='Alcohol Content', title='Alcohol content in different wine quality types')
# plt.show()
# #To-do
#
# plot2 = sns.boxplot(x="rating", y='sulphates', data = data)
# plot2.set(xlabel='Wine Ratings', ylabel='Sulphates', title='Sulphates vs Wine ratings')
# plt.show()
#To-Do
#Hyperparameter Tuning, Feature Selection, Scaling in Data Prepocessing, Class Imbalance, xgb,
# scores, rmsc mean square for regresssion, , psquare, accuracy, r2, 3-4 types of scores

#Report 2 Models
#Classifciation Model -1
#print(data.head(10))

data.drop('type', axis=1, inplace=True)
bins = (2, 6.5, 8)
classes = ['bad','good']
data['quality'] = pd.cut(data['quality'], bins = bins, labels = classes)
label_quality = LabelEncoder()
data['quality'] = label_quality.fit_transform(data['quality'].astype('str'))
data.info()

data.drop('rating', axis=1, inplace=True)
x = data.drop('quality', axis = 1)
y = data['quality']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.19, random_state = 21)
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
classy = RandomForestClassifier(n_estimators=50)
classy.fit(x_train, y_train)
y_pred = classy.predict(x_test)

classy = LogisticRegression()
classy.fit(x_train, y_train)
lof_pred = classy.predict(x_test)

# using metrics module for accuracy calculation
print("Accuracy of Random Forest model is: ", metrics.accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test,y_pred))


print("Accuracy of Logistic model is: ", metrics.accuracy_score(y_test, lof_pred))
print("Classification Report:")
print(classification_report(y_test,lof_pred))

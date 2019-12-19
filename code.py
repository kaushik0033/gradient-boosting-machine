# --------------
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
#Loading of the data
data=pd.read_csv(path)
data.head(5)
print("How many men and women (sex feature) are represented in this dataset : ",data.sex.value_counts())
print(" What is the average age (age feature) of women: ",data[data['sex']=='Female'].age.mean())
print("What is the percentage of German citizens (native-country feature) : ",data['native-country'].value_counts(normalize=True).loc['Germany']) 
print("For people who recieve more than 50K per year (salary feature), the mean and standard deviation of their age : ",data[data.salary=='>50k'].age.mean(),data[data.salary=='>50k'].age.std())
print("For people who recieve less than 50K per year (salary feature), the mean and standard deviation of their age : ",data[data.salary=='<=50k'].age.mean(),data[data.salary=='<=50k'].age.std())
print("The statistics of age for each gender for all the races : ",data.groupby(['race','sex']).age.describe())
filterBymaxvalue=data.groupby(['race','sex']).max().unstack().age
print("Maximum age of men of Amer-Indian-Eskimo race : ",filterBymaxvalue['Male']['Amer-Indian-Eskimo'])
#Encode salary for more or less than 50K
label_encoder = LabelEncoder() 
label_encoder=label_encoder.fit(['<=50k','>50k'])
data.salary=label_encoder.fit_transform(data.salary)
# get categorical featurers
cat_col=data.select_dtypes(include='object').columns.tolist()
# check unique categories in each column
for col in cat_col:
    print(col, " : ",len(data[col].unique())," labels")
#Create copy of original dataset
data1=data.copy()
# create a function to encode top x categories in each column otherwise it leads to increase of features,curse of diemntionality
def one_hot_top_x(df,x):
    for col_name in cat_col:
        for label in data1[col_name].value_counts().sort_values(ascending=False).head(x).index:
            df[col_name+'_'+label]=np.where(data1[col_name]==label,1,0)
#perform top 10 categories onehotencoding
one_hot_top_x(data1,10)
#drop categorical features now
data1.drop(cat_col,axis=1,inplace=True)
# create train,test and validation data
X=data1.drop(columns=['salary'],axis=1)
y=data1.salary
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
#create DecisionTreeClassifier 
clf_tree_gini=DecisionTreeClassifier(criterion='gini',random_state=1)
#clf_tree=DecisionTreeClassifier(criterion='gini',random_state=1,max_depth=5,min_samples_leaf=5)
#using giniindex method
clf_tree_gini.fit(X_train,y_train)
y_pred_gini=clf_tree_gini.predict(X_test)
cls_tree_gini_accuracyscore=accuracy_score(y_test,y_pred_gini)
print(cls_tree_gini_accuracyscore)
#using Entropy method
clf_tree_entropy=DecisionTreeClassifier(criterion='entropy',random_state=1)
clf_tree_entropy.fit(X_train,y_train)
y_pred_entropy=clf_tree_entropy.predict(X_test)
clf_tree_entropy_accuracyscore=accuracy_score(y_test,y_pred_entropy)
print(clf_tree_entropy_accuracyscore)
estimators=[]
estimators.append(('entropy',clf_tree_entropy))
logi_regression=LogisticRegression()
estimators.append(('logistic',logi_regression))
ensemble_models=VotingClassifier(estimators,voting='soft',)
ensemble_models.fit(X_train,y_train)
y_pred_ensemble=ensemble_models.predict(X_test)
accu_score_ensemble=accuracy_score(y_test,y_pred_ensemble)
print(accu_score_ensemble)
clf_random_1=RandomForestClassifier()
parameter_grid = {"n_estimators":[50],"max_depth": [6]}
grid_search_1=GridSearchCV(clf_random_1,parameter_grid)
grid_search_1.fit(X_train,y_train)
score_gs_ontest=grid_search_1.score(X_test,y_test)
score_gs_onval=grid_search_1.score(X_val,y_val)
print("Score on test :", score_gs_ontest,", Score on validation :",score_gs_onval," using random classifier")

parameter_grid_2 = {"n_estimators":[100],"max_depth": [6]}
grid_search_2=GridSearchCV(clf_random_1,parameter_grid_2)
grid_search_2.fit(X_train,y_train)
score_gs_2=grid_search_2.score(X_test,y_test)
score_gs_2_onval=grid_search_2.score(X_val,y_val)
print("Score on test :", score_gs_2,", Score on validation :",score_gs_2_onval," using random classifier")
# Train with GDB algorithm

#learning_rates = [0.05, 0.10, 0.25, 0.50, 0.75, 1]
#for learning_rate in learning_rates:
#    gb = GradientBoostingClassifier(n_estimators=50, learning_rate = learning_rate, max_depth = 6, random_state = 0)
#    gb.fit(X_train, y_train)
#    print("Learning rate: ", learning_rate)
#    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
#    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))
#    print()
gb = GradientBoostingClassifier(n_estimators=50, learning_rate = 0.05, max_depth = 6, random_state = 0)
gb.fit(X_train, y_train)
print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(gb.score(X_val, y_val)))
impotance_features = gb.feature_importances_
indices = np.argsort(impotance_features)[::-1]
top_k = 10
new_indices = indices[:top_k]   
# Print the feature ranking
print("Feature ranking:")

for f in range(top_k):
    print("%d. feature %d (%f)" %(f + 1, new_indices[f], impotance_features[new_indices[f]]))
#Plot the feature importances 
feat_importances = pd.Series(gb.feature_importances_, index=X_train.columns)
top10_f=feat_importances.nlargest(10)
top10_f.plot(kind='barh')

gb_50 = GradientBoostingClassifier(n_estimators=50, learning_rate = 0.10, max_depth = 6, random_state = 0)
gb_50.fit(X_train, y_train)
print("Accuracy score (training): {0:.3f}".format(gb_50.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(gb_50.score(X_val, y_val)))
gb_10 = GradientBoostingClassifier(n_estimators=10, learning_rate = 0.10, max_depth = 6, random_state = 0)
gb_10.fit(X_train, y_train)
print("Accuracy score (training): {0:.3f}".format(gb_10.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(gb_10.score(X_val, y_val)))
print(abs(gb_10.score(X_val, y_val)-gb_10.score(X_train, y_train)))
gb_100 = GradientBoostingClassifier(n_estimators=100, learning_rate = 0.10, max_depth = 6, random_state = 0)
gb_100.fit(X_train, y_train)
print("Accuracy score (training): {0:.3f}".format(gb_100.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(gb_100.score(X_val, y_val)))
print(abs(gb_100.score(X_val, y_val)-gb_100.score(X_train, y_train)))
train_err_50=gb_50.score(X_train, y_train)
train_err_10=gb_10.score(X_train, y_train)
train_err_100=gb_100.score(X_train, y_train)
training_errors=[train_err_10, train_err_50, train_err_100]
validation_err_50=gb_50.score(X_val, y_val)
validation_err_10=gb_10.score(X_val, y_val)
validation_err_100=gb_100.score(X_val, y_val)
validation_errors=[validation_err_10,validation_err_50,validation_err_100]

plt.plot([10, 50, 100], training_errors, linewidth=4.0, label='Training error')
plt.plot([10, 50, 100], validation_errors, linewidth=4.0, label='Validation error')
plt.xlabel("Number of trees")
plt.ylabel("Classification eror")
plt.title("Error vs number of trees")    
plt.show()





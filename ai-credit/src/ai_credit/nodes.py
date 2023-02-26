import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
    
def processing_data(dataset):
    dataset.rename({'b':'Gender', '30.83':'Age','0': 'Debt','u':'Married','g':'BankCustomer', 'w':'EducationLevel','v': 'Ethnicity','1.25':'YearsEmployed',
          't':'PriorDefault', 't.1':'Employed','01': 'CreditScore','f':'DriversLicense','g.1':'Citizen', '00202':'ZipCode','0.1': 'Income','+':'Approved'},axis=1,inplace=True)
    dataset.replace('?', np.NaN, inplace=True)
    dataset['Age']=dataset['Age'].astype(float)
    dataset['Approved'].replace(['+','-'],[1,0],inplace=True)
    dataset.to_csv('data/02_intermediate/Credit_Approval_rename_data.csv', index = False, header=True)
    # Impute the missing values with mean imputation
    dataset.fillna(dataset.mean(), inplace=True)
    # Iterate over each column of dataset
    for col in dataset:
        # Check if the column is of object type
        if dataset[col].dtypes == 'O':
            # Impute with the most frequent value
            dataset = dataset.fillna(dataset[col].value_counts().index[0])
    # list of numerical variables
    numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']
    discrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25 ]
    continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature]
    categorical_features=[feature for feature in dataset.columns if dataset[feature].dtypes=='O']
    return dataset, continuous_feature, categorical_features

def detect_outliers(data):
    outliers=[]
    threshold=3
    mean = np.mean(data)
    std =np.std(data)
    for i in range(len(data)):
        z_score= (data[i] - mean)/std 
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers

def processing_data_encode(data):
    df=pd.get_dummies(data[['Gender','Married','BankCustomer','PriorDefault','Employed','DriversLicense','Citizen']],drop_first=True)
    data=pd.concat([df,data],axis=1)
    data.drop(['Gender','Married','BankCustomer','PriorDefault','Employed','DriversLicense','Citizen'],axis=1,inplace=True)
    data['Approved']=data['Approved'].astype(int)
    education_level_encoding=data.groupby(['EducationLevel'])['Approved'].mean().to_dict()
    data['EducationLevel']=data['EducationLevel'].map(education_level_encoding)
    education_level_encoding=data.groupby(['Ethnicity'])['Approved'].mean().to_dict()
    data['Ethnicity']=data['Ethnicity'].map(education_level_encoding)
    ethnicity_encoding=data.groupby(['Ethnicity'])['Approved'].mean().to_dict()
    data['Ethnicity']=data['Ethnicity'].map(ethnicity_encoding)
    data.to_csv('data/03_primary/encode.csv', index = False, header=True)
    return data
    
def feature_selection(dataset):
    X=dataset.iloc[:,:-1]
    y=dataset['Approved']
    train_x = pd.get_dummies(X)
    ### Apply SelectKBest Algorithm
    ordered_rank_features=SelectKBest(score_func=chi2,k=15)
    ordered_feature=ordered_rank_features.fit(train_x,y)
    dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
    dfcolumns=pd.DataFrame(X.columns)
    features_rank=pd.concat([dfcolumns,dfscores],axis=1)
    features_rank.columns=['Features','Score']
    model=ExtraTreesClassifier()
    model.fit(train_x,y)
    ranked_features=pd.Series(model.feature_importances_,index=train_x.columns)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(train_x.values, i) for i in range(train_x.shape[1])]
    vif["features"] = train_x.columns
    from sklearn.feature_selection import mutual_info_classif
    mutual_info=mutual_info_classif(train_x,y)
    dataset[['PriorDefault_t', 'YearsEmployed','CreditScore','Income','Approved']]
    dataset.to_csv('data/04_feature/Feature_Selection.csv', index = False, header = True)
    return dataset
    

def train_test_split(dataset, parameters):
    #### Independent And Dependent features
    X=dataset.drop('Approved',axis=1)
    y=dataset['Approved']
    train_x = pd.get_dummies(X)
    #### Train Test Split
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(train_x,y,test_size = parameters["test_size"],random_state= parameters["random_state"])
    return X_train,X_test,y_train,y_test


def train(X_train,y_train, parameters):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import RandomizedSearchCV
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    classifier = DecisionTreeClassifier()
    random_grid= parameters["grid"]
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations
    model = RandomizedSearchCV(estimator = classifier, param_distributions = random_grid,scoring=parameters["scoring"], n_iter = parameters["n_iter"], cv = parameters["cv"], verbose= parameters["verbose"], random_state=parameters["random_state"], n_jobs = parameters["n_jobs"])
    model.fit(X_train,y_train)
    return model

def predict_on_test_data(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def predict_prob_on_test_data(model,X_test):
    y_pred_prob = model.predict_proba(X_test)
    return y_pred_prob

def get_metrics(y_test, y_pred, y_pred_prob):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred,average='micro')
    recall = recall_score(y_test, y_pred,average='micro')
    entropy = log_loss(y_test, y_pred_prob)
    metrics = {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)}
    return metrics


def create_confusion_matrix_plot(model, X_test, y_test):
    import matplotlib.pyplot as plt
    from sklearn.metrics import plot_confusion_matrix
    plot_confusion_matrix(model, X_test, y_test)
    plt.savefig('confusion_matrix.png')

def create_experiment(model, run_metrics, parameters, 
                      roc_auc_plot_path = None, run_params=None):
    import mlflow
    #mlflow.set_tracking_uri("http://localhost:5000") 
    #use above line if you want to use any database like sqlite as backend storage for model else comment this line
    mlflow.set_experiment("mlruns")
    mlflow.end_run()
    with mlflow.start_run(run_name="model_v1"):
        
        if run_params != None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])
            
        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])

        if parameters["cf_mt"] != None:
            mlflow.log_artifact(parameters["cf_mt"], 'confusion_matrix')
            
        if roc_auc_plot_path != None:
            mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")
        
        mlflow.set_tag("tag1", "Iris Classifier")
        mlflow.set_tags({"tag2":"Logistic Regression", "tag3":"Multiclassification using Ovr - One vs rest class"})
        mlflow.sklearn.log_model(model, "model", registered_model_name="model_v1")
        

def prediction(dataset_test):
    X_test=dataset_test.iloc[:,:-1]
    import mlflow.pyfunc

    model_name = "model_v1"
    model_version = 1

    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )
    x_test= pd.get_dummies(X_test)
    y_pred = model.predict(x_test)
    print(y_pred)

    sklearn_model = mlflow.sklearn.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )
    y_pred_prob = sklearn_model.predict_proba(x_test)
    print(y_pred_prob)
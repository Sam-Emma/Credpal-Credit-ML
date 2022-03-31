import pandas as pd
import numpy as np
dataset = pd.read_csv('Credit Default Clean Dataset.csv')

print("pass")
def preprocessing(data):
    #data['SALARY'] = data['SALARY'].replace(',', '')
    
    data['SALARY'] = data['SALARY'].astype(float)
    data['AVERAGE PAYMENT DELAY'] = data['AVERAGE PAYMENT DELAY'].replace(0,-1)
    data['AVERAGE PAYMENT DELAY'] = data['AVERAGE PAYMENT DELAY'].replace(-2,-1)
    #data = pd.get_dummies(data,columns = ['SEX','MARRIAGE'],prefix_sep = '-',drop_first=True)
    print("pass2")
    X = data[['SALARY', 'EDUCATION', 'AVERAGE PAYMENT DELAY']]
    return X


def predict(salary,education,avg_pay_delay):
    if (salary == "" or avg_pay_delay == ""):
        return "Fill in the missing field(s)", "Fill in the missing field(s)"
    else:
        X_dataframe = pd.DataFrame([{'SALARY':salary, 'EDUCATION':education,'AVERAGE PAYMENT DELAY':avg_pay_delay}])
        print(education)
        preprocessed_test_data = preprocessing(X_dataframe)
    
        prediction,confid = makePrediction(preprocessed_test_data)
        print("SUCCESS2")
        return prediction,confid 

def makePrediction(test_pre):
    X = preprocessing(dataset)
    y = dataset['DEFAULT']
    from sklearn.ensemble import RandomForestClassifier
    #from sklearn.metrics import f1_score,accuracy
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
    classifier = RandomForestClassifier()
    classifier.fit(X_train,y_train)
    confidence = 0
    predict_real = ""
    prediction = classifier.predict(test_pre)
    prob_prediction = classifier.predict_proba(test_pre)
    if (prob_prediction[0][1] < 50 and prediction[0] == 0):
        confidence = str(np.round(prob_prediction[0][0],2)*100)+"%"
    else:
        confidence = str(np.round(prob_prediction[0][1],2)*100)+"%"
    if(prediction[0] == 0):
        predict_real = "Unlikely to default"
    else:
        predict_real = "Likely to default"
    print("SUCCESS1")
    return predict_real, confidence
# -*- coding: utf-8 -*-


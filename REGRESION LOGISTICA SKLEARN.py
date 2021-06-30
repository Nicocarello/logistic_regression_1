
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('D:/Anaconda/datasets/bank/bank.csv',sep=';')

print(data)

print(data.columns.values)

#REEMPLAZO LOS YES/NO A 1/0

data['y'] = (data['y']=='yes').astype(int)

print(data['education'].unique())

#CAMBIO NOMBRES DE VALORES EN COLUMNA EDUCACION
data['education'] = np.where(data['education']=='basic.4y','Basic',data['education'])
data['education'] = np.where(data['education']=='basic.6y','Basic',data['education'])
data['education'] = np.where(data['education']=='basic.9y','Basic',data['education'])

data['education'] = np.where(data['education']=='high.school','High School',data['education'])
data['education'] = np.where(data['education']=='professional.course','Professional',data['education'])
data['education'] = np.where(data['education']=='university.degree','University',data['education'])
data['education'] = np.where(data['education']=='illiterate','Illiterate',data['education'])
data['education'] = np.where(data['education']=='unknown','Unknown',data['education'])

print(data)

#VEO CUANTAS PERSONAS COMPRARO O NO EL PRODUCTO

print(data['y'].value_counts())

#VEO EL PROMEDIO DE TODOS LOS DATOS SEGUN SI COMPRARON O NO

print(data.groupby('y').mean())

print(data.groupby('education').mean())

pd.crosstab(data.education, data.y).plot(kind='bar')
table = pd.crosstab(data.month, data.y)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)

#REEMPLAZO LAS VARIABLES CATEGORICAS POR NUMEROS

data.job.replace(('blue-collar', 'services', 'admin.', 'entrepreneur', 'self-employed',
 'technician', 'management', 'student', 'retired', 'housemaid', 'unemployed',
 'unknown'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True)

data.marital.replace(('married', 'single', 'divorced', 'unknown'),(1,2,3,4),inplace=True)
data.education.replace(('basic.9y', 'high.school', 'university.degree', 'professional.course',
 'basic.6y', 'basic.4y', 'unknown', 'illiterate'),(1,2,3,4,5,6,7,8),inplace=True)

data.default.replace(('no','yes','unknown'),(1,2,3),inplace=True)
data.housing.replace(('no','yes','unknown'),(1,2,3),inplace=True)
data.loan.replace(('no','yes','unknown'),(1,2,3),inplace=True)
data.contact.replace(('cellular','telephone'),(1,2),inplace=True)
data.month.replace(('mar','apr','may', 'jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10),inplace=True)
data.day_of_week.replace(('mon','tue','wed','thu','fri'),(1,2,3,4,5),inplace=True)
data.poutcome.replace(('nonexistent', 'failure', 'success'),(1,2,3),inplace=True)

#SELECCIONO LAS COLUMNAS QUE VOY A USAR EN EL MODELO

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#AGARRO TODAS LAS COLUMNAS DEL DATASET
bank_columns = data.columns.values.tolist()
#ASIGNO A Y EL NOMBRE DE LA COLUMNA QUE QUIERO PREDECIR
y=['y']
#ASIGNO A X TODAS LAS COLUMNAS MENOS LA QUE QUIERO PREDECIR
x=[column for column in bank_columns if column not in y]

x = data[x]
y = data[y]
lr = LogisticRegression()

rfe=RFE(lr,len(x))
rfe =rfe.fit(x,y.values.ravel())
print(rfe.score(x,y))
respuesta = rfe.predict(x)

#PRINTEO LA PRIORIDAD DE LAS COLUMNAS PARA PREDECIR
print(rfe.support_)
print(rfe.ranking_)

#IMPLEMENTACION DEL MODELO CON STATSMODEL

import statsmodels.api as sm

logit = sm.Logit(y,x)

result = logit.fit()
print(result.summary2())

from sklearn import linear_model

modelo=linear_model.LogisticRegression()
modelo.fit(x, y)
print(modelo.score(x,y))

df_rta = pd.DataFrame(
    {
     'prediccion': respuesta
     })
df_final = data.join(df_rta)

coeficientes = pd.DataFrame(list(zip(x.columns,np.transpose(modelo.coef_))))

##VALIDACION DEL MODELO LOGISTICO

#DIVIDO EL DATASET EN TRAIN Y TEST
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x, y,test_size=0.30,random_state = 0)

lm = linear_model.LogisticRegression()
lm.fit(x_train,y_train)

#CALCULO LAS PROBABILIDADES DE QUE UN USUARIO COMPRE O NO

probabilidades = lm.predict_proba(x_test)
print(probabilidades)

prediction = lm.predict(x_test)

prob = probabilidades[:,1]
prob_df = pd.DataFrame(prob)
threshold = 0.1
prob_df['prediccion'] = np.where(prob_df[0]>threshold,1,0)

a = pd.crosstab(prob_df.prediccion, columns='count')

from sklearn import metrics

#COMPARO LA PREDICCION CON LOS VALORES REALES
print(metrics.accuracy_score(y_test, prediction))

#VALIDACION CRUZADA/CROSS VALIDATION


from sklearn.model_selection import cross_val_score

#LAS PROBABILIDADES DE CADA REGISTRO DE PREDECIR SI COMPRA O NO
scores = cross_val_score(lm, x, y,scoring='accuracy',cv = 100)


#MATRICES DE CONFUSION Y CURVAS ROC

x_Train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
lm = linear_model.LogisticRegression()
lm.fit(x_train,y_train)

probs = lm.predict_proba(x_test)

prob = probs[:,1]
prob_df = pd.DataFrame(prob)
threshold = 0.1
prob_df['prediction'] = np.where(prob_df[0]>=threshold,1,0)
prob_df['actual'] = list(y_test)

confusion_matrix = pd.crosstab(prob_df.prediction, prob_df.actual)



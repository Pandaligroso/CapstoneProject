import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import argparse

#Crear el objeto ArgumentParser
parser = argparse.ArgumentParser(description='Ejemplo de validación cruzada con argumentos')

# Agregar los argumentos
parser.add_argument('--dataset', type=str, default='Datasets\weatherAUS.csv', help='Ruta del archivo del conjunto de datos')
parser.add_argument('--test_size', type=float, default=0.3, help='Proporción del conjunto de datos para usar como datos de prueba')
parser.add_argument('--random_state', type=int, default=27, help='Semilla aleatoria para dividir los datos de entrenamiento y prueba')

# Obtener los argumentos
args = parser.parse_args()

# Cargar el conjunto de datos y leer el archivo CSV
df = pd.read_csv(args.dataset)


# Inspeccionar el DataFrame
df.head(5)

# Obtener el número de filas y columnas del conjunto de datos
df.shape

# Estadísticas de todas las columnas del conjunto de datos
df.describe()

# Nombres de las columnas en el conjunto de datos
df.columns

df.nunique()

# Información del conjunto de datos
df.info()

# Convertir todas las columnas a minúsculas
df.columns = df.columns.str.strip().str.lower()
df.columns


df['date'] = pd.to_datetime(df['date']) # parse as datetime

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

df[['date', 'year', 'month', 'day']] # preview changes made

df.drop('date', axis=1, inplace=True)
df.info()

df.isnull().sum()

# Comprobar la completitud de los datos
missing = pd.DataFrame(df.isnull().sum(), columns=['no.of missing values'])

missing['% missing_values'] = (missing / len(df)).round(2) * 100
missing

df = df.drop(['sunshine', 'evaporation', 'cloud3pm', 'cloud9am'], axis=1)

missing

# Eliminar filas donde faltan las variables objetivo
df.dropna(how='all', subset=['raintomorrow'], inplace=True)

# Extraer características numéricas
num_col = df.select_dtypes(include=np.number).columns.to_list()
len(num_col)

df.head()

# Extraer características categóricas
cat_col = df.select_dtypes(object).columns.tolist()
len(cat_col)

for i in num_col:
    fig, axs = plt.subplots(1, 2, figsize=(15, 3))
    sns.histplot(df[i], bins=20, kde=True, ax=axs[0])
    sns.boxplot(df[i], ax=axs[1], color='#99befd', fliersize=1)

# Eliminar la columna 'rainfall' y las columnas numéricas
df = df.drop(['rainfall'], axis=1)
num_col.remove('rainfall')

# Imputar los valores faltantes de las características numéricas
median_values = df[num_col].median()
df[num_col] = df[num_col].fillna(value=median_values)

# Convertir los valores categóricos en valores numéricos
le = LabelEncoder()
df[cat_col] = df[cat_col].astype('str').apply(le.fit_transform)

# Imputar los valores faltantes de las características categóricas
mode_values = df[cat_col].mode()
df[cat_col] = df[cat_col].fillna(value=mode_values)

df.isnull().sum()

df.head()

# Eliminar las variables correlacionadas
df = df.drop(columns=['temp9am', 'temp3pm', 'pressure9am'], axis=1)

numcol_del = ['temp9am', 'temp3pm', 'pressure9am']
num_cols = list(set(num_col) - set(numcol_del))
num_cols

df.shape

cat_col

df.describe()

# Dividir los datos de entrenamiento y prueba
X = df.drop(['raintomorrow'], axis=1)
y = df['raintomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

# XGBoost
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)
xgb_y_pred = xgb_classifier.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
xgb_mae = mean_absolute_error(y_test, xgb_y_pred)
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_recall = recall_score(y_test, xgb_y_pred)
xgb_fmeasure = f1_score(y_test, xgb_y_pred)
print("Metrics for XGBoost:")
print("Accuracy:", xgb_accuracy)
print("MAE:", xgb_mae)
print("MSE:", xgb_mse)
print("Recall:", xgb_recall)
print("F-measure:", xgb_fmeasure)

from sklearn.decomposition import PCA

# Preprocesamiento de datos con PCA
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

pca = PCA(n_components=.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

import matplotlib.pyplot as plt
pd.DataFrame(pca.explained_variance_ratio_).plot.bar()
plt.legend('')
plt.xlabel('Principal Components')
plt.ylabel('Explained Varience');

# LSTM
X_train_lstm = np.reshape(X_train_pca, (X_train_pca.shape[0], X_train_pca.shape[1], 1))
X_test_lstm = np.reshape(X_test_pca, (X_test_pca.shape[0], X_test_pca.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_pca.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_data=(X_test_lstm, y_test))


lstm_y_pred = lstm_model.predict(X_test_lstm)
lstm_y_pred = (lstm_y_pred > 0.5)
lstm_accuracy = accuracy_score(y_test, lstm_y_pred)
lstm_mae = mean_absolute_error(y_test, lstm_y_pred)
lstm_mse = mean_squared_error(y_test, lstm_y_pred)
lstm_recall = recall_score(y_test, lstm_y_pred)
lstm_fmeasure = f1_score(y_test, lstm_y_pred)
print("Metrics for LSTM:")
print("Accuracy:", lstm_accuracy)
print("MAE:", lstm_mae)
print("MSE:", lstm_mse)
print("Recall:", lstm_recall)
print("F-measure:", lstm_fmeasure)

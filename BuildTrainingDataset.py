import pandas as pd
import numpy as np
from config import config
from sklearn.utils import shuffle

class BuildTrainingDataset(object):
    
    def __init__(self):
        self.df = None
        self.df_fraud = None
        self.df_normal = None
    
    def run(self):
        self.load_dataset()
        self.build_preprocessed_dataset()
        self.build_training_dataset()

    def load_dataset(self):
        self.df = pd.read_csv(config['dataset_source'])
    
    def build_preprocessed_dataset(self):
        # Eliminamos los features que tienen distribuciones muy similares entre los dos tipos de transacciones.
        # Obtenidas del proceso de exploración de datos.
        self.df = self.df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)

        # Creamos nuevos features para identificar valores donde las transacciones fraudulentas son más comunes
        self.df['V1_'] = self.df.V1.map(lambda x: 1 if x < -3 else 0)
        self.df['V2_'] = self.df.V2.map(lambda x: 1 if x > 2.5 else 0)
        self.df['V3_'] = self.df.V3.map(lambda x: 1 if x < -4 else 0)
        self.df['V4_'] = self.df.V4.map(lambda x: 1 if x > 2.5 else 0)
        self.df['V5_'] = self.df.V5.map(lambda x: 1 if x < -4.5 else 0)
        self.df['V6_'] = self.df.V6.map(lambda x: 1 if x < -2.5 else 0)
        self.df['V7_'] = self.df.V7.map(lambda x: 1 if x < -3 else 0)
        self.df['V9_'] = self.df.V9.map(lambda x: 1 if x < -2 else 0)
        self.df['V10_'] = self.df.V10.map(lambda x: 1 if x < -2.5 else 0)
        self.df['V11_'] = self.df.V11.map(lambda x: 1 if x > 2 else 0)
        self.df['V12_'] = self.df.V12.map(lambda x: 1 if x < -2 else 0)
        self.df['V14_'] = self.df.V14.map(lambda x: 1 if x < -2.5 else 0)
        self.df['V16_'] = self.df.V16.map(lambda x: 1 if x < -2 else 0)
        self.df['V17_'] = self.df.V17.map(lambda x: 1 if x < -2 else 0)
        self.df['V18_'] = self.df.V18.map(lambda x: 1 if x < -2 else 0)
        self.df['V19_'] = self.df.V19.map(lambda x: 1 if x > 1.5 else 0)
        self.df['V21_'] = self.df.V21.map(lambda x: 1 if x > 0.6 else 0)

        # Creamos nuevo feature para transacciones normales (No Fraude).
        self.df.loc[self.df.Class == 0, 'Normal'] = 1
        self.df.loc[self.df.Class == 1, 'Normal'] = 0

        # Renombramos el target 'Class' a 'Fraud'.
        self.df = self.df.rename(columns={'Class': 'Fraud'})

        # Obtenemos:
        # 492 transacciones fraudulentas (0.172% del total de transacciones normales)
        # 284,315 transacciones normales
        print("Transacciones Normales")
        print(self.df.Normal.value_counts())
        print("\n")
        print("Transacciones de Fraude")
        print(self.df.Fraud.value_counts())
        self.df.to_csv(config['dataset_preprocessed'])

    def build_training_dataset(self):
        print("Construyendo datasets de entrenamiento y prueba")
        # Creamos dataframes para transacciones fraudulentas y normales.
        self.df_fraud = self.df[self.df.Fraud == 1]
        self.df_normal = self.df[self.df.Normal == 1]

        # X_train = Variables predictoras para entrenamiento
        # X_test = Variables predictoras para pruebas

        # y_train = Variable target para entrenamiento
        # y_test = Variable target para pruebas

        # X_train = 80% de transacciones fraudulentas
        X_train = self.df_fraud.sample(frac=config['pct_data_train'])
        count_Frauds = len(X_train)

        # Agregamos 80% de transacciones normales a X_train.
        X_train = pd.concat([X_train, self.df_normal.sample(frac = 0.8)], axis = 0)

        # X_test contiene todas las transacciones que no están en X_train.
        X_test = self.df.loc[~self.df.index.isin(X_train.index)]

        # Combinamos los DF para que el entrenamiento se realice de manera aleatoria.
        X_train = shuffle(X_train)
        X_test = shuffle(X_test)

        # Agregamos la variable target a la data de entrenamiento y pruebas ("y_train","y_test").
        y_train = X_train.Fraud
        y_train = pd.concat([y_train, X_train.Normal], axis=1)

        y_test = X_test.Fraud
        y_test = pd.concat([y_test, X_test.Normal], axis=1)

        # Elinamos el target de las variables predictoras (X_train, X_test).
        X_train = X_train.drop(['Fraud','Normal'], axis = 1)
        X_test = X_test.drop(['Fraud','Normal'], axis = 1)

        # Verificamos la cantidad de registros de cada dataframe (train, test)
        print("Cantidad de registros")
        print("Dataset X Train: {0}".format(len(X_train))) # 227846
        print("Dataset X Test: {0}".format(len(X_test))) # 56961
        print("Dataset Y Train: {0}".format(len(y_train))) # 227846
        print("Dataset Y Test: {0}".format(len(y_test))) # 56961

        # Calculamos el ratio, debido a que contamos con una data desbalanceada.
        ratio = len(X_train)/count_Frauds 

        y_train.Fraud *= ratio
        y_test.Fraud *= ratio

        # Almacenamos el nombre de todas las variables predictoras X_train.
        features = X_train.columns.values

        # Transformamos cada feature para que tenga una media de 0 y una STD de 1 para
        # la fase de entrenamiento.
        for feature in features:
            mean, std = self.df[feature].mean(), self.df[feature].std()
            X_train.loc[:, feature] = (X_train[feature] - mean) / std
            X_test.loc[:, feature] = (X_test[feature] - mean) / std

        # Guardamos los dataset de entrenamiento y pruebas.
        print("Guardando datasets en disco.")
        X_train.to_csv(config['dataset_x_train'])
        X_test.to_csv(config['dataset_x_test'])
        y_train.to_csv(config['dataset_y_train'])
        y_test.to_csv(config['dataset_y_test'])
        print("Datasets de entrenamiento y prueba guardados!")
    
if __name__=="__main__":
    btd = BuildTrainingDataset()
    btd.run()
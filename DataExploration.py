import pandas as pd
import numpy as np
from config import config
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

class DataExploration(object):
    
    def __init__(self):
        self.df = None
    
    def run(self):
        self.load_dataset()
        self.show_info_dataset()
        self.show_describe_dataset()
        self.show_summary_target_by_time()
        self.show_chart_target_by_time()
        self.show_summary_target_by_amount()
        self.show_chart_target_by_amount()
        self.show_chart_target_by_time_amount()
        self.show_chart_histograms_anonymized_features()
    
    def load_dataset(self):
        self.df = pd.read_csv(config['dataset_source'])
    
    def show_info_dataset(self):
        print("\nESTRUCTURA DATASET\n")
        print(self.df.head())
        print("\n CANTIDAD DE REGISTROS: {0}".format(self.df.shape[0]))
        str(input())

    def show_describe_dataset(self):
        print("VALORES ESTADÍSTICOS DATASET\n")
        print(self.df.describe())
        str(input())

    def show_summary_target_by_time(self):
        """
        Comparación del tiempo entre transacciones fraudulentas y normales.
        """
        print ("\nCOMPARACIÓN POR TIEMPO DE TRANSACCIONES\n")
        print ("TRANSACCIONES DE FRAUDE")
        print (self.df.Time[self.df.Class == 1].describe())
        print ("\n")
        str(input())
        print ("TRANSACCIONES NORMALES")
        print (self.df.Time[self.df.Class == 0].describe())
        str(input())
    
    def show_chart_target_by_time(self):
        """
        Visualizando histograma por tipo (Fraude / No Fraude) basado en el tiempo de las transacciones.
        """
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
        bins = 50

        ax1.hist(self.df.Time[self.df.Class == 1], bins = bins)
        ax1.set_title('Fraude')

        ax2.hist(self.df.Time[self.df.Class == 0], bins = bins)
        ax2.set_title('Normal')

        plt.xlabel('Tiempo (en segundos)')
        plt.ylabel('Número de transacciones')
        plt.savefig('images/01_histograma_transacciones_tiempo.png')
        plt.show()
    
    def show_summary_target_by_amount(self):
        """
        Explorando si el monto de la transacción difiere entre los dos tipos de transacciones (Fraude / No Fraude)
        """
        print ("\nCOMPARACIÓN POR MONTO DE TRANSACCIONES\n")
        print ("TRANSACCIONES DE FRAUDE\n")
        print (self.df.Amount[self.df.Class == 1].describe())
        print ()
        print ("TRANSACCIONES NORMALES\n")
        print (self.df.Amount[self.df.Class == 0].describe())
        str(input())

    def show_chart_target_by_amount(self):
        """
        Visualizando histograma por tipo (Fraude / No Fraude) basado en el monto de las transacciones.
        """
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
        bins = 30

        ax1.hist(self.df.Amount[self.df.Class == 1], bins = bins)
        ax1.set_title('Fraude')

        ax2.hist(self.df.Amount[self.df.Class == 0], bins = bins)
        ax2.set_title('Normal')

        plt.xlabel('Monto ($)')
        plt.ylabel('Número de Transacciones')
        plt.yscale('log')
        plt.savefig('images/02_histograma_transacciones_monto.png')
        plt.show()

    def show_chart_target_by_time_amount(self):
        """
        Visualizando histograma por tipo (Fraude / No Fraude) basado en el tiempo y monto de las transacciones.
        """
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,6))

        ax1.scatter(self.df.Time[self.df.Class == 1], self.df.Amount[self.df.Class == 1])
        ax1.set_title('Fraude')

        ax2.scatter(self.df.Time[self.df.Class == 0], self.df.Amount[self.df.Class == 0])
        ax2.set_title('Normal')

        plt.xlabel('Tiempo (en segundos)')
        plt.ylabel('Monto')
        plt.savefig('images/03_histograma_transacciones_tiempo_monto.png')
        plt.show()
    
    def show_chart_histograms_anonymized_features(self):
        """
        Visualizando histogramas para cada feature anonimizado.
        """
        print("Generando histogramas")
        # Seleccionamos features anonimizados.
        v_features = self.df.iloc[:,1:29].columns
        plt.figure(figsize=(12,28*4))
        gs = gridspec.GridSpec(28, 1)
        for i, cn in enumerate(self.df[v_features]):
            ax = plt.subplot(gs[i])
            sns.histplot(self.df[cn][self.df.Class == 1], bins=50)
            sns.histplot(self.df[cn][self.df.Class == 0], bins=50)
            ax.set_xlabel('')
            ax.set_title('Histograma Feature: ' + str(cn))
        plt.savefig('images/04_histogramas_features.png')
        print("Histogramas de features guardados.")

if __name__=="__main__":
    da = DataExploration()
    da.run()
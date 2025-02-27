import pandas as pd


class DataProcesing:
    """
    Carga y preprocesa los datasets de usuarios, productos e interacciones 
    para su uso en el motor de recomendaciones.
    """

    def __init__(self):
        """
        Inicializa una instancia de DataProcesing.
        """
        self.PATH_DATASET = 'data/'

    def load_and_clean_data(self):
        """
        Carga los datos desde CSV y realiza una limpieza inicial.

        - Convierte timestamps a formato datetime.
        - (Opcional) Filtra productos sin stock disponible.

        Returns:
            tuple (DataFrame, DataFrame, DataFrame): (users, products, interactions)
        """
        print("Loading dataset...")
        # Cargar datos
        print("Loading dataset...")
        users = pd.read_csv(self.PATH_DATASET + 'users.csv')
        products = pd.read_csv(self.PATH_DATASET + 'products.csv', sep=';')
        interactions = pd.read_csv(self.PATH_DATASET + 'interactions.csv')

        # Convertir timestamp a datetime
        interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])

        # Seleccionar solo productos con stock disponible
        #products = products[products['stock_actual'] > 0]

        return users, products, interactions
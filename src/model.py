import numpy as np
from sklearn.neighbors import NearestNeighbors


class Model:
    """

    """

    def __init__(self):
        """
        Inicializa una instancia de Model.
        """
        self.PATH_DATASET = 'data/'

    def train_knn_model(self, user_product_matrix):
        # Convierte a array denso y reemplaza NaN con 0
        dense_matrix = user_product_matrix.toarray()
        dense_matrix = np.nan_to_num(dense_matrix)

        print(f"Cantidad de NaN en matriz antes de entrenar: {np.isnan(dense_matrix).sum()}")  # Depuración
        
        model = NearestNeighbors(metric="cosine", algorithm="brute")
        model.fit(dense_matrix)
        return model

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from constants.constants import keyword_to_interest


class Utils:
    """
    Clase de utilidades para la preprocesamiento de datos en un sistema de recomendación.
    Incluye funciones para el procesamiento de usuarios, productos e interacciones.
    """

    def __init__(self):
        """
        Inicializa una instancia de Utils.
        """
        self.PATH_DATASET = 'data/'

    def process_user_features(self, users, interactions):
        """
        Procesa las características de los usuarios, aplicando escalado, codificación
        y representación de intereses.

        Args:
            users (pd.DataFrame): Datos de los usuarios.
            interactions (pd.DataFrame): Historial de interacciones usuario-producto.

        Returns:
            pd.DataFrame, pd.DataFrame: Matrices de características de usuarios e intereses.
        """
        mapeo_frecuencia = {'Diaria': 1, 'Semanal': 1/7, 'Mensual': 1/30}
        users['frecuencia_login'] = users['frecuencia_login'].map(mapeo_frecuencia)
        
        mapeo_nivel_ingresos = {'Alto': 3, 'Medio': 2, 'Bajo': 1}
        users['nivel_ingresos'] = users['nivel_ingresos'].map(mapeo_nivel_ingresos)
        
        encoder = OneHotEncoder(sparse_output=False)
        user_features = users[['genero', 'tipo_suscripcion', 'categoria_cliente', 'ubicacion', 'dispositivo', 'nivel_educativo']]
        encoded_features = encoder.fit_transform(user_features)
        
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(users[['edad', 'nivel_ingresos', 'frecuencia_login']])
        
        user_features_matrix = np.hstack([scaled_features, encoded_features])
        user_features_df = pd.DataFrame(user_features_matrix, index=users['user_id'])
        user_features_df = user_features_df.loc[interactions['user_id'].unique()]
        
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
        user_interest_matrix = vectorizer.fit_transform(users['intereses'])
        user_interest_df = pd.DataFrame(user_interest_matrix.toarray(), 
                                        index=users['user_id'], 
                                        columns=vectorizer.get_feature_names_out())
        
        return user_features_df, user_interest_df

    def map_product_keywords_to_interests(self, product_vector):
        """
        Mapea palabras clave de productos a categorías de interés predefinidas.

        Args:
            product_vector (pd.Series): Vector binario de palabras clave del producto.

        Returns:
            pd.Series: Vector de categorías de interés con conteo acumulado.
        """
        interest_vector = {interest: 0 for interest in set(keyword_to_interest.values())}
        for keyword, value in product_vector.items():
            if value == 1 and keyword in keyword_to_interest:
                interest_vector[keyword_to_interest[keyword]] += 1
        return pd.Series(interest_vector)

    def build_user_product_matrix(self, interactions, products, user_features, user_interest_df):
        """
        Construye la matriz de interacciones usuario-producto y la matriz de características
        de productos basada en palabras clave.

        Args:
            interactions (pd.DataFrame): Historial de interacciones usuario-producto.
            products (pd.DataFrame): Datos de los productos disponibles.
            user_features (pd.DataFrame): Características procesadas de los usuarios.
            user_interest_df (pd.DataFrame): Matriz de intereses de los usuarios.

        Returns:
            tuple: Matriz dispersa de usuario-producto, índices de usuario y producto,
                   y matriz de intereses de productos.
        """
        user_idx = {user: idx for idx, user in enumerate(interactions['user_id'].unique())}
        product_idx = {product: idx for idx, product in enumerate(products['product_id'].unique())}
        
        interactions['rating'] = interactions.apply(lambda row: row['rating'] if row['tipo_interaccion'] == 'Valoracion' else 1, axis=1)
        
        rows = interactions['user_id'].map(user_idx).dropna().astype(int).values
        cols = interactions['product_id'].map(product_idx).dropna().astype(int).values
        data = interactions['rating'].values
        
        user_product_matrix = coo_matrix((data, (rows, cols)), shape=(len(user_idx), len(product_idx)))
        user_feature_matrix = csr_matrix(user_features.loc[user_idx.keys()].values)
        
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
        product_matrix = vectorizer.fit_transform(products['palabras_clave'])
        product_df = pd.DataFrame(product_matrix.toarray(), 
                                index=products['product_id'], 
                                columns=vectorizer.get_feature_names_out())
        
        product_interest_df = product_df.apply(self.map_product_keywords_to_interests, axis=1).fillna(0)
        
        full_user_matrix = hstack([user_product_matrix.tocsr(), user_feature_matrix])
        
        return full_user_matrix, user_idx, product_idx, product_interest_df

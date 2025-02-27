from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Recomender:
    """
    Implementa un sistema de recomendación basado en similitud de intereses 
    entre usuarios y productos hibrido utilizando la similitud del coseno.
    """

    def __init__(self):
        """
        Inicializa una instancia de Recomender.
        """

    def recommend_products_by_interest(self, user_id, user_interest_df, product_interest_df, products, top_n=5):
        """
        Genera recomendaciones de productos basadas en los intereses del usuario.

        Args:
            user_id (int): ID del usuario al que se le harán recomendaciones.
            user_interest_df (pd.DataFrame): Matriz de intereses de los usuarios.
            product_interest_df (pd.DataFrame): Matriz de características de los productos.
            products (pd.DataFrame): Información de los productos (ID, nombre, categoría).
            top_n (int): Número de recomendaciones a devolver (por defecto 5).

        Returns:
            list[dict]: Lista de productos recomendados con 'product_id', 'name' y 'category'.
        """
        if user_id not in user_interest_df.index:
            return []
        
        user_idx = user_interest_df.index.get_loc(user_id)
        similarity_matrix = cosine_similarity(user_interest_df, product_interest_df)
        user_similarities = similarity_matrix[user_idx]
        recommended_product_indices = np.argsort(user_similarities)[::-1][:top_n]
        
        recommendations = products.iloc[recommended_product_indices][['product_id', 'name', 'category']]
        return recommendations.to_dict(orient='records')

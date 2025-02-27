from flask import Flask, request, jsonify
from src.data_procesing import DataProcesing
from src.utils import Utils
from src.model import Model
from src.recomender import Recomender
import json

app = Flask(__name__)

dp = DataProcesing()
users, products, interactions = dp.load_and_clean_data()

utils = Utils()

user_features, user_interest_df = utils.process_user_features(users, interactions)
user_product_matrix, user_idx, product_idx, product_interest_df = utils.build_user_product_matrix(interactions, products, user_features, user_interest_df)

model = Model()
knn_model = model.train_knn_model(user_product_matrix)

rc = Recomender()

#user_id = 113
#interest_recommendations = rc.recommend_products_by_interest(user_id, user_interest_df, product_interest_df, products, top_n=5)

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id', type=int)
    if user_id is None:
        return jsonify({"error": "Falta el parámetro user_id"}), 400
    
    recommendations = rc.recommend_products_by_interest(user_id, user_interest_df, product_interest_df, products, top_n=5)
    
    return app.response_class(
        response=json.dumps({"user_id": user_id, "recommendations": recommendations}, ensure_ascii=False, indent=2),
        mimetype="application/json"
    )

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=5000, debug=True)

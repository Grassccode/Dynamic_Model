import pandas as pd
import pickle
from flask import Flask, request, jsonify
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

# Load model jika sudah ada, kalau tidak buat baru nanti
try:
    with open('model/mba.pkl', 'rb') as f:
        market_basket_model = pickle.load(f)
except FileNotFoundError:
    market_basket_model = None

# Endpoint untuk memuat data dan menjalankan Market Basket Analysis
@app.route('/load_data', methods=['POST'])
def load_data():
    try:
        # Mengambil file CSV dari request
        file = request.files['file']
        df = pd.read_csv(file)

        # Terapkan algoritma Apriori untuk mencari frequent itemsets
        frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)

        # Mendapatkan association rules dari frequent itemsets
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        #print(rules)

        # Update model yang sudah dimuat sebelumnya
        global market_basket_model
        market_basket_model = rules

        # Ambil product set dari request (dalam format JSON)
        data = market_basket_model
        #print(product_set)
        
        # Konversi frozenset ke dalam list agar bisa di-serialize ke JSON
        data['antecedents'] = data['antecedents'].apply(lambda x: list(x))
        data['consequents'] = data['consequents'].apply(lambda x: list(x))

        # Konversi hasilnya ke dalam format JSON yang sesuai
        result = data[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_dict(orient='records')
        print(result)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Main entry point
if __name__ == '__main__':
    app.run(port=5001, debug=True)

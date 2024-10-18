import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from opensearchpy import OpenSearch, OpenSearchException
from flask import Flask, jsonify, request

def get_sample_data(url):
    response = requests.get(url)
    encoding = response.encoding if response.encoding else 'utf-8'
    content_string = response.content.decode(encoding)
    sample_data = [json.loads(bulk_request) for i, bulk_request in enumerate(content_string.split('\n')) if (i + 1) % 2 == 0]
    return sample_data

def enhance_semantic_vectors(matrix, features):
    enhanced_matrix = matrix.toarray().copy()
    for i, doc_vector in enumerate(enhanced_matrix):
        words = [features[col] for col, value in enumerate(doc_vector) if value > 0]
        empty_vectors = {features[col]: col for col, value in enumerate(doc_vector) if value == 0}
        semantic_vector = calculate_similarity_score(empty_vectors, words)
        if len(semantic_vector) > 0:
            for vec in semantic_vector:
                doc_vector[vec[0]] = vec[1]
            enhanced_matrix[i] = doc_vector
    return normalize(enhanced_matrix)

def calculate_similarity_score(empty_vectors, words):
    semantic_apple_products = {
        'macbook': {'iphone': 0.0005, 'ipad': 0.0003},
        'iphone': {'ipad': 0.0005, 'macbook': 0.0005},
        'ipad': {'iphone': 0.0005, 'macbook': 0.0003}
    }

    semantic_vector = []
    for word in words:
        if word in semantic_apple_products:
            for key in semantic_apple_products[word].keys():
                if key in empty_vectors: 
                    semantic_vector.append((empty_vectors[key], 
                                            semantic_apple_products[word][key]))
    return semantic_vector


def create_vector_index(client, index_name):
    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100
            }
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "text"
                },
                "description": {
                    "type": "text"
                },
                "vector": {
                    "type": "knn_vector",
                    "dimension": 3041,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 24
                        }
                    }
                }
            }
        }
    }

    try:
        # Check if the index already exists
        if not client.indices.exists(index=index_name):
            response = client.indices.create(index=index_name, body=index_body)
            print(f"Index '{index_name}' created successfully:")
            print(response)
        else:
            print(f"Index '{index_name}' already exists. Skipping creation.")
    except OpenSearchException as e:
        # This will catch any OpenSearch specific exceptions
        print(f"An OpenSearch error occurred: {str(e)}")
    except Exception as e:
        # This will catch any other exceptions
        print(f"An error occurred: {str(e)}")

def store_vector_to_opensearch(client, matrix_with_semantic_feature):
    bulk_data = []
    for index, vector in enumerate(matrix_with_semantic_feature):
        bulk_data.append(json.dumps({
            "update": {
                "_index": "product",
                "_id": index
            }
        }))
        
        bulk_data.append(json.dumps({
            "doc": {'vector': list(vector)},
            "doc_as_upsert": True
        }))
    body = '\n'.join(bulk_data) + '\n'
    response = client.bulk(body)
    
    success_count = sum(1 for item in response['items'] if item['update']['status'] in [200, 201])
    failed_count = len(response['items']) - success_count
    
    print(f"Successfully updated {success_count} documents")
    if failed_count > 0:
        print(f"Failed to update {failed_count} documents")

def run_vector_search(client, vector, size, k=100):
    query = {
        "size": size,
        "query": {
            "script_score": {
                "query": {
                    "knn": {
                        "vector": {
                            "vector": vector,
                            "k": k
                        }
                    }
                },
                "script": {
                    "source": "_score > 0.5 ? _score : 0"
                }
            }
        },
        "min_score": 0.5,
        "_source": ["title", "description"]
    }

    try:
        response = client.search(index='product', body=query)
        return response
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


# define a variable
host = 'localhost'
port = 9200
client = OpenSearch(hosts=[{'host': host, 'port': port}])
gist_url = url = 'https://gist.githubusercontent.com/sog01/1fb9e3ad87198a54a48b8a4e5a47c9b5/raw/21f72f47e9ab00ca512015ffdf59f730abff1341/amazon-products-sample.json'

# create a vector index
create_vector_index(client, 'product')

# define a TF-IDF vectorizer
sample_data = get_sample_data(url)
s_data = [data['title']+'\n'+data['description'] for data in sample_data]
vectorizer = TfidfVectorizer().fit(s_data)

app = Flask(__name__)

@app.route('/load-data')
def loadData():
    # load data sample to OpenSearch
    response = requests.get(gist_url)
    encoding = response.encoding if response.encoding else 'utf-8'
    body = response.content.decode(encoding)
    client.bulk(body)

    # load data vector to OpenSearch
    matrix = vectorizer.transform(s_data)
    matrix_with_semantic_feature = enhance_semantic_vectors(matrix, vectorizer.get_feature_names_out())
    store_vector_to_opensearch(client, matrix_with_semantic_feature)

    return jsonify({'message': 'success load data'}), 200

@app.route('/vector-search')
def search():
    q = request.args.get('q', '')
    size = request.args.get('size', 3)
    if q == '':
        return jsonify({'error': 'query cannot be empty'}), 400
    vector_query = vectorizer.transform([q])
    search_result = run_vector_search(client, vector_query.toarray()[0], size)
    return jsonify(search_result)

@app.route('/')
def topSearch():
    response = client.search(index='product', body={"_source": ["title", "description"]})
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
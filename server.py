import os
import logging
import numpy as np
import torch
import resource
from flask import Flask, request, jsonify, send_from_directory
from transformers import DistilBertTokenizer, DistilBertModel
import faiss

app = Flask(__name__)
app.config['STATIC_FOLDER'] = '.'

# Configure logging
logging.basicConfig(level=logging.ERROR, filename='error.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Global variables
base_dir = 'repos'
code_snippets = []
file_names = []
file_extensions = []
index = None
chunk_size = 100  # Reduced chunk size
memory_limit_gb = 7 # Adjust this based on your system's memory

def extract_code_files(chunk_num, chunk_size, total_files):
    global code_snippets, file_names, file_extensions
    code_snippets = []
    file_names = []
    file_extensions = []
    start_index = chunk_num * chunk_size
    end_index = min((chunk_num + 1) * chunk_size, len(all_files))
    processed_files = 0

    for i in range(start_index, end_index):
        file_path = all_files[i]
        processed_files += 1
        print(f"Adding file to index: {file_path} ({processed_files}/{total_files})")
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.py', '.js', '.java', '.cpp', '.rb', '.go','.ts', '.tsx', '.html', '.css', '.scss', '.c', '.h', '.m', '.swift', '.rs', '.cs', '.go', '.php', '.sql', '.pl', '.sh', '.r', '.tex', '.md', '.json', '.yaml', '.yml', '.xml', '.config', '.ini', '.properties', '.cfg', '.txt', '.log', '.csv', '.tsv', '.dat', '.conf', '.dockerfile', '.makefile', '.bat', '.ps1', '.sh', '.rb', '.vue', '.dart']:  # Extend this list as needed
            continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_snippets.append(f.read())
                file_names.append(file_path)
                file_extensions.append(ext)
        except Exception as e:
            logging.exception(f"Error reading {file_path}: {e}")

def get_embeddings(snippets):
    inputs = tokenizer(snippets, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return np.array(embeddings).astype('float32')

def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    # Use HNSW for faster search and better memory efficiency
    index = faiss.IndexHNSWFlat(d, 32) # 32 is the number of connections in the graph
    index.add(embeddings)
    return index

def check_memory():
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_gb = mem / (1024**3)
    if mem_gb > memory_limit_gb * 0.9:
        raise MemoryError("Memory limit exceeded")

all_files = []
for root, _, files in os.walk(base_dir):
    if 'node_modules' in root or '.git' in root:
        continue
    for file in files:
        all_files.append(os.path.join(root, file))

@app.route('/refresh', methods=['POST'])
def refresh_index():
    global index
    index = None
    total_files = len(all_files)
    try:
        num_chunks = (len(all_files) + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            extract_code_files(i, chunk_size, total_files)
            check_memory()
            embeddings = get_embeddings(code_snippets)
            if index is None:
                index = build_faiss_index(embeddings)
            else:
                index.add(embeddings)
        return jsonify({'message': 'Index refreshed.'})
    except MemoryError as e:
        logging.exception(f"Memory error during index building: {e}")
        return jsonify({'message': 'Memory error during index building'}), 500
    except Exception as e:
        logging.exception(f"Error refreshing index: {e}")
        return jsonify({'message': 'Error refreshing index'}), 500


@app.route('/search', methods=['POST'])
def search_code():
    if index is None:
        return jsonify({'message': 'Index not yet built. Please call /refresh first.'}), 200

    query = request.json.get('query')
    k = request.json.get('k', 5)

    try:
        query_embedding = get_embeddings([query])
        query_embedding = query_embedding.reshape(1, -1) # Correct reshape
        print(f"query_embedding shape: {query_embedding.shape}")
        D, I = index.search(query_embedding, k)

        results = [
            {
                'file_name': file_names[i],
                'file_extension': file_extensions[i],
                'code_snippet': code_snippets[i][:200],  # Return only the first 200 chars for brevity
                'distance': float(D[0][j])
            }
            for j, i in enumerate(I[0])
        ]
        return jsonify({'results': results, 'total_results': len(results)})
    except Exception as e:
        logging.exception(f"Error during search: {e}")
        return jsonify({'message': 'Error during search'}), 500

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True, port=5100)

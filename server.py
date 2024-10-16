import os
import numpy as np
import torch
import resource
from flask import Flask, request, jsonify, send_from_directory
from transformers import DistilBertTokenizer, DistilBertModel
import faiss

app = Flask(__name__)
app.config['STATIC_FOLDER'] = '.'

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
index_file_path = 'faiss_index.bin'

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
        app.logger.info(f"Adding file to index: {file_path} ({processed_files}/{total_files})")
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.py', '.js', '.java', '.cpp', '.rb', '.go','.ts', '.tsx', '.html', '.css', '.scss', '.c', '.h', '.m', '.swift', '.rs', '.cs', '.go', '.php', '.sql', '.pl', '.sh', '.r', '.tex', '.md', '.json', '.yaml', '.yml', '.xml', '.config', '.ini', '.properties', '.cfg', '.txt', '.log', '.csv', '.tsv', '.dat', '.conf', '.dockerfile', '.makefile', '.bat', '.ps1', '.sh', '.rb', '.vue', '.dart']:  # Extend this list as needed
            continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_snippets.append(f.read())
                file_names.append(file_path)
                file_extensions.append(ext)
        except Exception as e:
            app.logger.exception(f"Error reading {file_path}: {e}")

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

def build_and_load_index():
    global index
    try:
        total_files = len(all_files)
        num_chunks = (total_files + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            extract_code_files(i, chunk_size, total_files)
            check_memory()
            embeddings = get_embeddings(code_snippets)
            if index is None:
                index = build_faiss_index(embeddings)
            else:
                index.add(embeddings)
        faiss.write_index(index, index_file_path)
        app.logger.info('Index built and saved.')
    except Exception as e:
        app.logger.exception(f"Error building index: {e}")

@app.route('/refresh', methods=['POST'])
def refresh_index():
    build_and_load_index()
    return jsonify({'message': 'Index refreshed and saved.'})


@app.route('/search', methods=['POST'])
def search_code():
    global index
    if index is None:
        if os.path.exists(index_file_path):
            try:
                index = faiss.read_index(index_file_path)
                app.logger.info("Index loaded successfully from file.")
            except Exception as e:
                app.logger.warning(f"Could not load index from file.  Attempting to rebuild. Error: {e}")
                build_and_load_index()
        else:
            build_and_load_index()

    if index is None:
        return jsonify({'message': 'Index not found. Please call /refresh first.'}), 200

    query = request.json.get('query')
    k = request.json.get('k', 5)

    try:
        query_embedding = get_embeddings([query])
        query_embedding = query_embedding.reshape(1, -1)
        D, I = index.search(query_embedding, k)

        results = []
        for j, i in enumerate(I[0]):
            if i < len(file_names):
                results.append({
                    'file_name': file_names[i],
                    'file_extension': file_extensions[i],
                    'code_snippet': code_snippets[i][:200],
                    'distance': float(D[0][j])
                })

        return jsonify({'results': results, 'total_results': len(results)})
    except Exception as e:
        app.logger.exception(f"Error during search: {e}")
        return jsonify({'message': 'Error during search'}), 500

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    build_and_load_index()
    app.run(debug=True, port=5100)

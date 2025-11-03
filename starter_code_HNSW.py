import faiss
import h5py
import numpy as np
import os
import urllib.request

def evaluate_hnsw():

    # start your code here
    # download data, build index, run query

    # Create data directory if it doesn't exist
    ROOT = os.path.dirname(__file__)
    DATA_DIR = os.path.join(ROOT, "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download dataset if not present
    HDF5_URL = "https://ann-benchmarks.com/sift-128-euclidean.hdf5"
    HDF5_PATH = os.path.join(DATA_DIR, "sift-128-euclidean.hdf5")
    
    if not os.path.exists(HDF5_PATH):
        print(f"Downloading {HDF5_URL} (~501MB)…")
        urllib.request.urlretrieve(HDF5_URL, HDF5_PATH)
        print("Download complete.")
    
    # Load vectors from dataset
    with h5py.File(HDF5_PATH, "r") as f:
        keys = set(f.keys())
        train_key = "train" if "train" in keys else ("base" if "base" in keys else None)
        test_key = "test" if "test" in keys else ("query" if "query" in keys else None)
        
        if train_key is None or test_key is None:
            raise KeyError(f"Unexpected HDF5 keys: {keys}")
        
        xb = f[train_key][:].astype("float32")  # Base/train vectors for indexing
        xq = f[test_key][:].astype("float32")   # Test/query vectors
    
    # Get dimension
    d = xb.shape[1]
    if d != 128:
        print(f"Warning: expected 128 dims, got {d}")
    
    # Build HNSW index with M=16
    M = 16
    index = faiss.IndexHNSWFlat(d, M)
    
    # Set efConstruction before adding vectors
    efConstruction = 200
    index.hnsw.efConstruction = efConstruction
    
    # Add vectors to index
    index.add(xb)
    
    # Set efSearch before searching
    efSearch = 200
    index.hnsw.efSearch = efSearch
    
    # Perform query using first query vector, search for top-10
    D, I = index.search(xq[0:1], 10)
    
    # write the indices of the 10 approximate nearest neighbours in output.txt, separated by new line in the same directory
    OUT_PATH = os.path.join(ROOT, "output.txt")
    with open(OUT_PATH, "w") as f:
        for idx in I[0]:
            f.write(f"{int(idx)}\n")
    
    print(f"Top-10 indices written to {OUT_PATH}")

if __name__ == "__main__":
    evaluate_hnsw()

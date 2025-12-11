import os
import sys
from typing import List, Dict, Any

# Add current directory to path so we can import modules from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from langchain_community.vectorstores import FAISS
from indexing import LCEmbeddings


def retrieve_by_id(rid: str, query: str, index_root: str = "faiss_indexes", k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve documents for a given ID and query using FAISS index.
    """
    # Construct the path to the specific index
    # indexing.py saves it as: os.path.join(index_root, f"id_{rid}")
    index_path = os.path.join(index_root, f"id_{rid}")
    
    if not os.path.exists(index_path):
        print(f"Warning: Index path does not exist: {index_path}")
        return []

    try:
        # Initialize embeddings
        embeddings = LCEmbeddings()
        
        # Load the FAISS index
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        
        # Perform similarity search
        docs = vectorstore.similarity_search(query, k=k)
        
        # Format results
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        return results
        
    except Exception as e:
        print(f"Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    # Test configuration
    test_id = "s_0qONh1o8EYcWAz00047f3JxUiC8eWo"
    # Using a generic query for testing
    test_query = "main topic" 
    
    found_root = "faiss_indexes"
    
    results = retrieve_by_id(test_id, test_query, index_root=found_root)
    print(f"Retrieved {len(results)} documents")

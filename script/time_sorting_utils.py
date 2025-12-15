from typing import List, Dict, Any
from .time_processor import enhance_retrieval_with_time_sorting, TimeProcessor

def sort_documents_by_time(documents: List[Dict[str, Any]], 
                          sort_by: str = 'earliest') -> List[Dict[str, Any]]:
    if not documents:
        return documents
    
    if isinstance(documents[0], dict) and 'dates' in documents[0]:
        processor = TimeProcessor()
        return processor.sort_documents_by_time(documents, sort_by)
    
    enhanced_docs = enhance_retrieval_with_time_sorting(documents, "")
    return enhanced_docs

def get_sorted_documents_with_time_info(documents: List[Dict[str, Any]], 
                                       sort_by: str = 'earliest') -> Dict[str, Any]:
    sorted_docs = sort_documents_by_time(documents, sort_by)
    
    docs_with_time = [doc for doc in sorted_docs if doc.get('has_time_info', False)]
    docs_without_time = [doc for doc in sorted_docs if not doc.get('has_time_info', False)]
    
    return {
        'sorted_documents': sorted_docs,
        'time_statistics': {
            'total_documents': len(sorted_docs),
            'documents_with_time': len(docs_with_time),
            'documents_without_time': len(docs_without_time),
            'time_coverage_ratio': len(docs_with_time) / len(sorted_docs) if sorted_docs else 0
        }
    }
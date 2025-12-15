import os
import sys
from typing import List, Dict, Any

# Add current directory to path so we can import modules from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from langchain_community.vectorstores import FAISS
from indexing import LCEmbeddings
from time_processor import enhance_retrieval_with_time_sorting


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


def retrieve_with_time_sorting(rid: str, query: str, index_root: str = "faiss_indexes", 
                              k: int = 5, sort_by: str = 'earliest') -> List[Dict[str, Any]]:
    """
    检索文档并按时间排序
    
    Args:
        rid: 文档ID
        query: 查询文本
        index_root: 索引根目录
        k: 检索数量
        sort_by: 排序方式 ('earliest' 或 'latest')
    
    Returns:
        按时间排序的文档列表
    """
    # 先进行普通检索
    chunks = retrieve_by_id(rid, query, index_root, k)
    
    if not chunks:
        return []
    
    # 应用时间排序增强
    sorted_docs = enhance_retrieval_with_time_sorting(chunks, query, sort_by)
    
    return sorted_docs


def get_time_statistics(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    获取时间统计信息
    
    Args:
        documents: 文档列表
        
    Returns:
        时间统计信息
    """
    docs_with_time = [doc for doc in documents if doc.get('has_time_info')]
    docs_without_time = [doc for doc in documents if not doc.get('has_time_info')]
    
    # 提取所有日期
    all_dates = []
    for doc in docs_with_time:
        dates = doc.get('dates', [])
        all_dates.extend(dates)
    
    # 计算时间范围
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        date_range = (max_date - min_date).days
    else:
        min_date = max_date = None
        date_range = 0
    
    return {
        'total_documents': len(documents),
        'documents_with_time': len(docs_with_time),
        'documents_without_time': len(docs_without_time),
        'time_coverage_ratio': len(docs_with_time) / len(documents) if documents else 0,
        'date_range_days': date_range,
        'earliest_date': min_date.isoformat() if min_date else None,
        'latest_date': max_date.isoformat() if max_date else None
    }


if __name__ == "__main__":
    # Test configuration
    test_id = "s_0qONh1o8EYcWAz00047f3JxUiC8eWo"
    # Using a generic query for testing
    test_query = "养发" 
    
    found_root = "faiss_indexes"
    
    # 测试普通检索
    print("=== 普通检索结果 ===")
    results = retrieve_by_id(test_id, test_query, index_root=found_root)
    print(f"检索到 {len(results)} 个文档分块")
    
    # 测试时间排序检索
    print("\n=== 时间排序检索结果 ===")
    time_sorted_results = retrieve_with_time_sorting(test_id, test_query, index_root=found_root, sort_by='earliest')
    print(f"时间排序后得到 {len(time_sorted_results)} 个文档")
    
    # 显示时间统计信息
    stats = get_time_statistics(time_sorted_results)
    print(f"\n=== 时间统计信息 ===")
    print(f"总文档数: {stats['total_documents']}")
    print(f"有时间信息的文档: {stats['documents_with_time']}")
    print(f"无时间信息的文档: {stats['documents_without_time']}")
    print(f"时间覆盖率: {stats['time_coverage_ratio']:.2%}")
    
    if stats['earliest_date']:
        print(f"最早日期: {stats['earliest_date']}")
        print(f"最晚日期: {stats['latest_date']}")
        print(f"时间跨度: {stats['date_range_days']} 天")
    
    # 显示前几个文档的时间信息
    print(f"\n=== 前3个文档的时间信息 ===")
    for i, doc in enumerate(time_sorted_results[:3]):
        print(f"文档 {i+1}:")
        print(f"  时间排名: {doc.get('time_rank')}")
        print(f"  是否有时间信息: {doc.get('has_time_info')}")
        if doc.get('earliest_date'):
            print(f"  最早时间: {doc.get('earliest_date')}")
        if doc.get('latest_date'):
            print(f"  最晚时间: {doc.get('latest_date')}")
        print(f"  分块数量: {doc.get('chunk_count')}")
        print()
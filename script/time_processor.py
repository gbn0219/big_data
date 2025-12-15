import re
import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class TimeProcessor:    
    def __init__(self):
        self.date_patterns = [
            # YYYY-MM-DD
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',
            # YYYY/MM/DD
            r'\b\d{4}/\d{1,2}/\d{1,2}\b',
            # YYYY年MM月DD日
            r'\b\d{4}年\d{1,2}月\d{1,2}日\b',
            # MM/DD/YYYY
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            # YYYY.MM.DD
            r'\b\d{4}\.\d{1,2}\.\d{1,2}\b',
            # 中文日期格式
            r'\b\d{4}年\d{1,2}月\b',
            r'\b\d{1,2}月\d{1,2}日\b',
            # 时间戳（10位或13位）
            r'\b\d{10}\b',
            r'\b\d{13}\b',
        ]
        
    def extract_dates_from_text(self, text: str) -> List[datetime.datetime]:
        dates = []
        
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                date_obj = self._parse_date(match)
                if date_obj:
                    dates.append(date_obj)
        
        return dates
    
    def _parse_date(self, date_str: str) -> Optional[datetime.datetime]:
        try:
            if re.match(r'\d{4}-\d{1,2}-\d{1,2}', date_str):
                return datetime.datetime.strptime(date_str, '%Y-%m-%d')
            
            elif re.match(r'\d{4}/\d{1,2}/\d{1,2}', date_str):
                return datetime.datetime.strptime(date_str, '%Y/%m/%d')
            
            elif re.match(r'\d{4}年\d{1,2}月\d{1,2}日', date_str):
                date_str = date_str.replace('年', '-').replace('月', '-').replace('日', '')
                return datetime.datetime.strptime(date_str, '%Y-%m-%d')
            
            elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_str):
                return datetime.datetime.strptime(date_str, '%m/%d/%Y')
            
            elif re.match(r'\d{10}', date_str):
                timestamp = int(date_str)
                # 检查时间戳是否在合理范围内（1970年-2100年）
                if timestamp > 0 and timestamp < 4102444800:  # 2100-01-01
                    return datetime.datetime.fromtimestamp(timestamp)
                else:
                    return None
                
            elif re.match(r'\d{13}', date_str):
                timestamp = int(date_str) / 1000
                # 检查时间戳是否在合理范围内（1970年-2100年）
                if timestamp > 0 and timestamp < 4102444800:  # 2100-01-01
                    return datetime.datetime.fromtimestamp(timestamp)
                else:
                    return None
                    
        except (ValueError, OverflowError, OSError) as e:
            logger.debug(f"无法解析日期: {date_str}, 错误: {e}")
            return None
        
        return None
    
    def get_earliest_date(self, dates: List[datetime.datetime]) -> Optional[datetime.datetime]:
        if not dates:
            return None
        return min(dates)
    
    def get_latest_date(self, dates: List[datetime.datetime]) -> Optional[datetime.datetime]:
        if not dates:
            return None
        return max(dates)
    
    def group_chunks_by_document(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        doc_groups = defaultdict(list)
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            doc_id = metadata.get('doc_id')
            if doc_id:
                doc_groups[doc_id].append(chunk)
        
        return dict(doc_groups)
    
    def merge_chunks_for_document(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {}
        
        merged_content = ' '.join([chunk.get('content', '') for chunk in chunks])
        
        all_dates = []
        for chunk in chunks:
            content = chunk.get('content', '')
            dates = self.extract_dates_from_text(content)
            all_dates.extend(dates)
        
        metadata = chunks[0].get('metadata', {}) if chunks else {}
        
        return {
            'content': merged_content,
            'metadata': metadata,
            'dates': all_dates,
            'earliest_date': self.get_earliest_date(all_dates),
            'latest_date': self.get_latest_date(all_dates),
            'chunk_count': len(chunks)
        }
    
    def sort_documents_by_time(self, documents: List[Dict[str, Any]], 
                              sort_by: str = 'earliest') -> List[Dict[str, Any]]:        
        def get_sort_key(doc):
            if sort_by == 'earliest':
                date = doc.get('earliest_date')
            else:  # sort_by == 'latest'
                date = doc.get('latest_date')
            
            # 如果没有日期信息，放到最后
            if date is None:
                return datetime.datetime.max
            return date
        
        return sorted(documents, key=get_sort_key)

def enhance_retrieval_with_time_sorting(retrieved_chunks: List[Dict[str, Any]], 
                                       query: str,
                                       sort_by: str = 'earliest') -> List[Dict[str, Any]]:
    processor = TimeProcessor()
    
    doc_groups = processor.group_chunks_by_document(retrieved_chunks)
    
    merged_docs = []
    for doc_id, chunks in doc_groups.items():
        merged_doc = processor.merge_chunks_for_document(chunks)
        if merged_doc:
            merged_docs.append(merged_doc)
    
    sorted_docs = processor.sort_documents_by_time(merged_docs, sort_by)
    
    for i, doc in enumerate(sorted_docs):
        doc['time_rank'] = i + 1
        doc['has_time_info'] = doc.get('earliest_date') is not None
    
    return sorted_docs
"""Event summary subagent."""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph

from context import Context
from model import get_response
from state import State

import re
import json
import ast

from retrieval_by_id import retrieve_by_id
from time_processor import enhance_retrieval_with_time_sorting
from langchain_community.vectorstores import FAISS
from indexing import LCEmbeddings

def event_summary_agent() -> StateGraph:
    """Event summary subagent."""
    builder = StateGraph(
        State,
        context_schema=Context,
    )
    # Nodes
    builder.add_node("generate_topic_and_query", generate_topic_and_query)
    builder.add_node("retrieve_and_rerank", retrieve_and_rerank)
    builder.add_node("generate_answer", generate_answer)

    # Edges
    builder.add_edge(START, "generate_topic_and_query")
    builder.add_edge("generate_topic_and_query", "retrieve_and_rerank")
    builder.add_edge("retrieve_and_rerank", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder


def generate_topic_and_query(
    state: State,
) -> dict:
    """Retireve and generate forecasting query."""
    task_id = state.get("task_id")
    
    # get topic
    MIN_CONTENT_LEN = 200
    example_doc_content = ""
    docs = retrieve_by_id(task_id, "", index_root="faiss_indexes")
    doc_num = 0
    max_doc_content_len = 0
    max_doc_content_id = 0
    for i in range(0, len(docs)):
        doc = docs[i]
        if doc_num == 2:
            break
        if len(doc["content"]) > max_doc_content_len:
            max_doc_content_len = len(doc["content"])
            max_doc_content_id = i
        if len(doc["content"]) > MIN_CONTENT_LEN:
            example_doc_content += doc["content"] + "\n\n"
            doc_num += 1

    if example_doc_content == "":
        example_doc_content += docs[max_doc_content_id].content

    prompt = (
        "你是一个经验丰富的新闻主题提取专家，负责提取给定新闻文字的主题。\n"
        "我将为你提供一段新闻文字，请提取新闻的主题，大约10个字。\n"
        "请严格按照以下 json 格式返回结果，不要添加任何额外的解释或代码块标记。例如：\n"
        "    {example}\n\n"
        "请提取以下文档的主题：{doc_content}\n"
    )
    example = {
        "topic": "大约10个字的主题"
    }
    
    prompt = prompt.format(
        doc_content=example_doc_content,
        example=example,
    )
    topic = get_response([{"role": "user", "content": prompt}])
    if topic.startswith("```json"):
        topic = topic[topic.find("\n")+1:topic.rfind("\n")]

    try:
        topic_json = json.loads(topic.replace("'", '/"'))
    except Exception:
        topic_json = ast.literal_eval(topic)

    topic = topic_json["topic"]


    prompt = (
        "你是一个经验丰富的提问词生成专家，负责为给定的新闻主题生成查询词。\n"
        "我将为你提供一个新闻主题，请返回所有可能的查询词。\n"
        "说明：\n"
        "    我需要为这个主题的新闻稿件写一个摘要，现在已经为稿件内容做好分块和向量索引。\n"
        "    请你为我生成一些能够提高检索结果可靠性的查询词或短语或句子，比如某某事件在何时何地发生等，\n"
        "    以便于索引检索得到的高相似度分块能包含事件尽可能多的主要信息。\n"
        "    我需要5个左右查询词或短语或句子，每个20字以内\n"
        "请严格按照以下 json 格式返回结果，不要添加任何额外的解释或代码块标记。例如：\n"
        "    {example}\n\n"
        "请提供查询词：{topic}\n"
    )
    example = {
        "query_words": [
            "查询词句1",
            "查询词句2",
            "查询词句3",
            "查询词句4",
            "查询词句5",
        ]
    }
    
    prompt = prompt.format(
        topic=topic,
        example=example,
    )
    query_words = get_response([{"role": "user", "content": prompt}])
    if query_words.startswith("```json"):
        query_words = query_words[query_words.find("\n")+1:query_words.rfind("\n")]

    try:
        query_words_json = json.loads(query_words.replace("'", '/"'))
    except Exception:
        query_words_json = ast.literal_eval(query_words)

    query_words = query_words_json["query_words"]

    return {
        "event_summary": {
            "query_words": query_words,
            "topic": topic,
            "docs": [],
            "summary": ""
        }
    }


def sort_documents_by_time(documents: list, sort_by: str = 'earliest') -> list:
    if not documents:
        return documents
    
    if isinstance(documents[0], dict) and 'dates' in documents[0]:
        from ..time_processor import TimeProcessor
        processor = TimeProcessor()
        return processor.sort_documents_by_time(documents, sort_by)
    
    enhanced_docs = enhance_retrieval_with_time_sorting(documents, "")
    return enhanced_docs

def rerank_documents_by_similarity_simple(documents: list, query: str, k: int = 5) -> list:
    if not documents or not query:
        return documents[:k] if documents else []
    
    import difflib
    
    scored_docs = []
    for doc in documents:
        content = doc.get('content', '')
        similarity = difflib.SequenceMatcher(None, query.lower(), content.lower()).ratio()
        scored_docs.append((similarity, doc))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # 返回前k个文档
    return [doc for score, doc in scored_docs[:k]]

def retrieve_and_rerank(
    state: State
):
    task_id = state.get("task_id")
    query_words = state.get("event_summary")["query_words"]

    docs_all = []
    query = ""
    for query_word in query_words:
        docs = retrieve_by_id(task_id, query_word, index_root="faiss_indexes")
        docs_all.extend(docs)
        query += query_word + " "  # 合并所有查询词
    
    # 去重（基于文档内容）
    unique_docs = []
    seen_contents = set()
    for doc in docs_all:
        content = doc.get('content', '')
        if content not in seen_contents:
            seen_contents.add(content)
            unique_docs.append(doc)
    
    if unique_docs:
        docs_reranked_by_similarity = rerank_documents_by_similarity_simple(
            unique_docs, query.strip(), k=5
        )
        
        docs_sorted_by_time = sort_documents_by_time(docs_reranked_by_similarity, sort_by='earliest')
        
        top_docs = docs_sorted_by_time
    else:
        top_docs = []

    return {
        "event_summary": {
            "query_words": query_words,
            "topic": state.get("event_summary")["topic"],
            "docs": top_docs,
            "summary": ""
        }
    }


def generate_answer(
    state: State
) -> str:
    """Generate answer."""
    topic = state.get("event_summary")["topic"]
    docs = state.get("event_summary")["docs"]
    
    # 提取文档内容
    doc_contents = []
    for doc in docs:
        if isinstance(doc, dict):
            content = doc.get('content', '')
        else:
            content = str(doc)
        doc_contents.append(content)
    
    prompt = (
        "你是一个经验丰富的事件摘要生成专家，负责生成“{topic}”主题的事件摘要。\n"
        "请严格基于以下提供的背景信息，生成一个简洁而准确的摘要，概括这些文档所描述事件的主要信息。事件已按照时间顺序排好序。\n\n"
        "要求：\n"
        "    1. 必须按照时间顺序，结合多个文档信息，生成完整的摘要。\n"
        "    2. 通常包含事件发生的时间、地点、主体，事件的起因、经过、结果等。\n"
        "    3. 总体字数在200字左右。\n"
        "背景信息：\n"
        "    {doc_content}\n\n"
        "请严格按照以下 json 格式返回结果，不要添加任何额外的解释或代码块标记。例如：\n"
        "    {example}\n\n"
    )
    example = {
        "summary": "摘要",
    }
    prompt = prompt.format(
        topic=topic,
        doc_content=";\n".join(doc_contents),
        example=example,
    )

    answer = get_response([{"role": "user", "content": prompt}])
    if answer.startswith("```json"):
        answer = answer[answer.find("\n")+1:answer.rfind("\n")]
    try:
        answer_json = json.loads(answer.replace("\'", "\""))
    except Exception:
        answer_json = ast.literal_eval(answer)
    
    print(answer_json["summary"])

    return {
        "event_summary": {
            "query_words": state.get("event_summary")["query_words"],
            "topic": topic,
            "docs": docs,
            "summary": answer_json["summary"]
        }
    }
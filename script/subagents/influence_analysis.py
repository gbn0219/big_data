"""Influence analysis subagent."""
from typing import Any, List, Dict, Optional

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph

from script.context import Context
from script.model import get_response
from script.state import State



def generate_influence_oriented_queries(topic: str) -> List[str]:
    """
    基于事件主题生成面向「影响分析」的检索Query（多维度扩展）
    核心：覆盖事件本身、相关方、影响领域、核心动作，提升检索文档的相关性
    """
    if not topic:
        return []

    # 基础核心Query（必选）
    core_queries = [
        topic,  # 原始主题
        f"{topic} 影响",  # 直接关联影响
        f"{topic} 后果",  # 同义词扩展
    ]

    # 影响维度扩展Query（覆盖分析所需的核心维度）
    dimension_queries = [
        f"{topic} 经济影响",
        f"{topic} 行业影响",
        f"{topic} 社会影响",
        f"{topic} 政策影响",
        f"{topic} 企业回应",
        f"{topic} 市场反应",
        f"{topic} 长期影响",
        f"{topic} 短期影响",
    ]

    # 去重并返回（避免重复检索）
    all_queries = list(set(core_queries + dimension_queries))
    return all_queries


def merge_retrieved_docs(task_id: int, topic: str, index_root: str = "faiss_indexes") -> str:
    """
    基于生成的多维度Query检索文档，并合并去重
    返回格式：拼接后的可读新闻内容（便于融入Prompt）
    """
    # 生成面向影响分析的检索Query
    queries = generate_influence_oriented_queries(topic)
    if not queries:
        return ""

    retrieved_docs = set()  # 用集合去重
    for query in queries:
        try:
            # 调用外部检索函数（按单个Query检索）
            doc_content = retrieve_by_id(task_id, query, index_root=index_root)
            if doc_content:
                # 拆分单条文档（假设retrieve_by_id返回多条文档用分隔符拼接，需根据实际调整）
                # 若retrieve_by_id直接返回单文本，可直接add
                for doc in doc_content.split("doc_id:"):
                    if doc.strip():
                        retrieved_docs.add(doc.strip())
        except Exception as e:
            # 单个Query检索失败不影响整体，仅日志记录
            continue

    # 拼接去重后的文档（格式化便于LLM读取）
    if retrieved_docs:
        merged_content = "\n\n".join([f"【相关资料{idx+1}】{doc}" for idx, doc in enumerate(retrieved_docs)])
        return merged_content
    return ""


def analyze_influence(state: State) -> State:
    """基于事件摘要+智能检索新闻+主题分析影响"""
    topic = state["topic"]
    task_id = state.get("task_id", "")

    # 1. 生成Query并检索新闻
    try:
        retrieved_news_content = merge_retrieved_docs(task_id, topic)
    except Exception as e:
        retrieved_news_content = ""
        state["history"].append({
            "role": "system",
            "content": f"检索新闻失败：{str(e)}"
        })

    # 2. 提取事件摘要
    try:
        event_summary_content = state.get("event_summary", {}).get("summary", "")
    except Exception as e:
        event_summary_content = ""
        state["history"].append({
            "role": "system",
            "content": f"提取事件摘要失败：{str(e)}"
        })

    # 3. 优化Prompt：明确LLM基于检索文档分析影响
    system_prompt = """你是专业的新闻事件影响分析专家，需基于提供的事件摘要、相关资料和事件主题，客观、全面地分析事件的影响。
分析要求：
1. 区分正面/负面影响，清晰说明影响的范围和程度；
2. 语言流畅、逻辑连贯，无需分维度标注，直接以自然文本输出分析内容；
3. 所有分析结论优先基于提供的相关资料和事件摘要；
4. 避免冗余，聚焦核心影响，无需重复事件本身，直接分析影响。
"""

    # 拼接Prompt：整合所有信息（按优先级：主题→摘要→检索文档）
    prompt_parts = [f"【分析主题】{topic}"]
    if event_summary_content:
        prompt_parts.append(f"\n【事件摘要】\n{event_summary_content}")
    if retrieved_news_content:
        prompt_parts.append(f"\n【参考资料】\n{retrieved_news_content}")
    else:
        prompt_parts.append("\n【参考资料】无相关文档")

    user_prompt = f"""请分析以下事件的影响：

{"".join(prompt_parts)}

输出要求：直接输出分析得到的影响的文本，无需任何额外说明、标题或格式。"""

    # 4. 调用LLM生成分析结果
    messages = [
        SystemMessage(content=system_prompt),
        {"role": "user", "content": user_prompt}
    ]

    try:
        # 调用LLM接口
        influence_content = get_response(messages)

        # 存储结果（保持字典结构）
        state["influence_analysis"] = {
            "event_topic": topic,
            "id": task_id,
            "influence_content": influence_content.strip()
        }
        state["history"].append({
            "role": "assistant",
            "content": "成功完成事件影响分析"
        })

    except Exception as e:
        # 异常处理：保证state字段完整性
        state["influence_analysis"] = {
            "event_topic": topic,
            "error": f"影响分析执行失败：{str(e)}",
            "influence_content": ""
        }
        state["history"].append({
            "role": "assistant",
            "content": f"影响分析失败：{str(e)}"
        })

    return state


def influence_analysis_agent() -> StateGraph:
    """构建影响分析子代理的状态图"""
    builder = StateGraph(
        State,
        context_schema=Context,
    )

    # 添加节点：仅保留分析节点（检索逻辑已内嵌）
    builder.add_node("analyze_influence", analyze_influence)

    # 定义执行流程
    builder.add_edge(START, "analyze_influence")
    builder.add_edge("analyze_influence", END)

    return builder
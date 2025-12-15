"""Answer merge subagent."""
from typing import Any

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph

from context import Context
from model import get_response
from state import State


def merge_answer(state: State) -> State:
    """
    调用LLM合并事件摘要和影响分析内容
    核心：保留摘要核心信息+影响分析的核心结论，合并为连贯文本
    """
    # 1. 从State中提取核心内容（增加容错，避免字段缺失）
    try:
        # 提取事件摘要（优先取summary字段）
        event_summary = state.get("event_summary", {})
        summary_content = event_summary.get("summary", "") if isinstance(event_summary, dict) else str(event_summary)
        summary_content = summary_content.strip()

        # 提取影响分析内容（优先取influence_content字段）
        influence_analysis = state.get("influence_analysis", {})
        influence_content = influence_analysis.get("influence_content", "") if isinstance(influence_analysis, dict) else str(influence_analysis)
        influence_content = influence_content.strip()

    except Exception as e:
        summary_content = ""
        influence_content = ""
        state["history"].append({
            "role": "system",
            "content": f"提取摘要/影响分析内容失败：{str(e)}"
        })

    # 2. 构造合并Prompt（明确LLM合并要求）
    system_prompt = """你是专业的文本整合专家，需将「事件摘要」和「事件影响分析」两部分内容合并为一段逻辑连贯、结构完整的文本。
合并要求：
1. 保留事件摘要的核心信息；
2. 保留事件影响分析的核心结论；
3. 无需额外新增内容，仅对两部分内容做有机整合，语言流畅、过渡自然；
"""

    # 拼接用户Prompt（区分空值场景）
    prompt_parts = []
    if summary_content:
        prompt_parts.append(f"【事件摘要】\n{summary_content}")
    if influence_content:
        prompt_parts.append(f"\n【事件影响分析】\n{influence_content}")

    # 无内容时直接返回空，避免调用LLM
    if not prompt_parts:
        state["answer_merge"] = {
            "merged_content": "",
            "status": "failed",
            "reason": "事件摘要和影响分析均为空，无法合并"
        }
        state["history"].append({
            "role": "assistant",
            "content": "合并失败：无可用的摘要/影响分析内容"
        })
        return state

    user_prompt = f"""请合并以下内容：

{"".join(prompt_parts)}

输出要求：直接输出合并后的文本，无需任何额外说明、标题或格式。"""

    # 3. 调用LLM执行合并
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        # 调用LLM接口获取合并结果
        merged_content = get_response(messages)
        merged_content = merged_content.strip()

        # 存储合并结果（结构化字段，便于后续使用）
        state["answer_merge"] = {
            "merged_content": merged_content,
            "has_summary": bool(summary_content),
            "has_influence": bool(influence_content)
        }
        state["history"].append({
            "role": "assistant",
            "content": "成功合并事件摘要和影响分析内容"
        })

    except Exception as e:
        # 异常处理：保证字段完整性
        state["answer_merge"] = {
            "merged_content": "",
            "reason": f"合并失败：{str(e)}",
            "has_summary": bool(summary_content),
            "has_influence": bool(influence_content)
        }
        state["history"].append({
            "role": "assistant",
            "content": f"合并失败：{str(e)}"
        })

    return state


def answer_merge_agent() -> StateGraph:
    """Answer merge subagent（合并摘要+影响分析）"""
    builder = StateGraph(
        State,
        context_schema=Context,
    )

    # 添加节点：合并摘要和影响分析
    builder.add_node("merge_answer", merge_answer)

    # 定义执行流程：启动 → 合并 → 结束
    builder.add_edge(START, "merge_answer")
    builder.add_edge("merge_answer", END)

    return builder
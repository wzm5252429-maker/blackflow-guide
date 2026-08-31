import os
import json
import argparse
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# -------------------- 1. 定义状态 Schema --------------------
class AgentState(TypedDict):
    document: str
    criteria: str
    agent1_report: Optional[Dict]
    agent2_report: Optional[Dict]
    agent3_report: Optional[Dict]
    discussion_history: List[Dict]
    current_round: int
    max_rounds: int
    converged: bool
    final_report: Optional[Dict]

# -------------------- 2. 初始化模型 --------------------
# 请确保已在环境变量中设置 OPENAI_API_KEY
model_agent1 = ChatOpenAI(model="gpt-4o", temperature=0.2)
model_agent2 = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)
model_agent3 = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# -------------------- 3. 节点函数定义（保持不变） --------------------
def create_independent_agent(agent_name: str, model, state_key: str):
    def node(state: AgentState) -> AgentState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个严谨的文档校验专家。请根据给定的校验标准，独立检查文档，不要参考其他意见。"),
            ("human",
             "文档内容：\n{document}\n\n"
             "校验标准：\n{criteria}\n\n"
             "请输出结构化报告，包含问题列表。每个问题包括：位置、描述、严重性（高/中/低）、建议修正、置信度。"
             "以JSON格式返回，格式示例：{{\"issues\": [{{\"位置\": \"...\", \"描述\": \"...\", \"严重性\": \"高\", \"建议修正\": \"...\", \"置信度\": 0.9}}]}}")
        ])
        chain = prompt | model
        response = chain.invoke({
            "document": state["document"],
            "criteria": state["criteria"]
        })
        try:
            report = json.loads(response.content)
        except json.JSONDecodeError:
            report = {"raw": response.content, "issues": []}
        state[state_key] = report
        return state
    return node

agent1_node = create_independent_agent("agent1", model_agent1, "agent1_report")
agent2_node = create_independent_agent("agent2", model_agent2, "agent2_report")
agent3_node = create_independent_agent("agent3", model_agent3, "agent3_report")

def discussion_round(state: AgentState) -> AgentState:
    reports = {
        "agent1": state["agent1_report"],
        "agent2": state["agent2_report"],
        "agent3": state["agent3_report"],
    }
    history = state.get("discussion_history", [])
    current_round = state.get("current_round", 0)
    new_reports = {}
    for agent_name, model in [("agent1", model_agent1), ("agent2", model_agent2), ("agent3", model_agent3)]:
        other_reports = {k: v for k, v in reports.items() if k != agent_name}
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个文档校验专家。你已经完成了自己的独立校验，现在请参考其他专家的报告，"
                      "指出你同意或反对的问题，并更新你的最终问题列表。只保留你认为确实存在的问题。"),
            ("human",
             "原始文档：\n{document}\n\n"
             "你的原始报告：\n{my_report}\n\n"
             "其他专家的报告：\n{other_reports}\n\n"
             "历史讨论记录：\n{history}\n\n"
             "请以JSON格式输出你的更新后报告，格式与之前相同。")
        ])
        chain = prompt | model
        response = chain.invoke({
            "document": state["document"],
            "my_report": reports[agent_name],
            "other_reports": other_reports,
            "history": history
        })
        try:
            updated_report = json.loads(response.content)
        except json.JSONDecodeError:
            updated_report = reports[agent_name]
        new_reports[agent_name] = updated_report

    state["agent1_report"] = new_reports["agent1"]
    state["agent2_report"] = new_reports["agent2"]
    state["agent3_report"] = new_reports["agent3"]
    state["discussion_history"] = history + [{"round": current_round, "reports": new_reports}]
    state["current_round"] = current_round + 1
    state["converged"] = (state["current_round"] >= state["max_rounds"])
    return state

def aggregate(state: AgentState) -> AgentState:
    all_issues = []
    for key in ["agent1_report", "agent2_report", "agent3_report"]:
        report = state[key]
        if isinstance(report, dict) and "issues" in report:
            for issue in report["issues"]:
                issue_with_source = issue.copy()
                issue_with_source["source"] = key
                all_issues.append(issue_with_source)
    unique_issues = {}
    for issue in all_issues:
        key = (issue.get("位置", ""), issue.get("描述", ""))
        if key not in unique_issues:
            unique_issues[key] = issue
            unique_issues[key]["支持者"] = [issue["source"]]
        else:
            unique_issues[key]["支持者"].append(issue["source"])
    final_report = {
        "issues": list(unique_issues.values()),
        "converged": state.get("converged", False),
        "discussion_rounds": state["current_round"],
    }
    state["final_report"] = final_report
    return state

# -------------------- 4. 构建状态图（保持不变） --------------------
graph = StateGraph(AgentState)
graph.add_node("start", lambda state: state)
graph.add_node("agent1", agent1_node)
graph.add_node("agent2", agent2_node)
graph.add_node("agent3", agent3_node)
graph.add_node("discussion", discussion_round)
graph.add_node("aggregate", aggregate)

graph.set_entry_point("start")
graph.add_edge("start", "agent1")
graph.add_edge("start", "agent2")
graph.add_edge("start", "agent3")
graph.add_edge("agent1", "discussion")
graph.add_edge("agent2", "discussion")
graph.add_edge("agent3", "discussion")

def should_continue_discussion(state: AgentState) -> str:
    if state["converged"] or state["current_round"] >= state["max_rounds"]:
        return "aggregate"
    else:
        return "discussion"

graph.add_conditional_edges(
    "discussion",
    should_continue_discussion,
    {"discussion": "discussion", "aggregate": "aggregate"}
)
graph.add_edge("aggregate", END)

app = graph.compile()

# -------------------- 5. 主程序：从文件读取文档和校验标准 --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多 Agent 文档校验系统")
    parser.add_argument("--document", "-d", required=True, help="待校验文档的文件路径（txt）")
    parser.add_argument("--criteria", "-c", required=True, help="校验标准文件路径（txt）")
    parser.add_argument("--max-rounds", "-r", type=int, default=2, help="最大讨论轮数（默认2）")
    args = parser.parse_args()

    # 读取文件内容
    with open(args.document, "r", encoding="utf-8") as f:
        document_text = f.read()
    with open(args.criteria, "r", encoding="utf-8") as f:
        criteria_text = f.read()

    # 初始化状态
    initial_state = {
        "document": document_text,
        "criteria": criteria_text,
        "agent1_report": None,
        "agent2_report": None,
        "agent3_report": None,
        "discussion_history": [],
        "current_round": 0,
        "max_rounds": args.max_rounds,
        "converged": False,
        "final_report": None,
    }

    print("开始运行多 Agent 校验流程...")
    final_state = app.invoke(initial_state)

    print("\n===== 最终校验报告 =====")
    if final_state["final_report"]:
        report = final_state["final_report"]
        print(f"讨论轮次: {report['discussion_rounds']}")
        print(f"是否收敛: {report['converged']}")
        print(f"发现的问题数量: {len(report['issues'])}")
        for i, issue in enumerate(report['issues'], 1):
            print(f"\n问题 {i}:")
            print(f"  位置: {issue.get('位置', '未知')}")
            print(f"  描述: {issue.get('描述', '未知')}")
            print(f"  严重性: {issue.get('严重性', '未知')}")
            print(f"  建议修正: {issue.get('建议修正', '无')}")
            print(f"  置信度: {issue.get('置信度', '未知')}")
            print(f"  支持者: {issue.get('支持者', [])}")
    else:
        print("未能生成最终报告，请检查运行日志。")
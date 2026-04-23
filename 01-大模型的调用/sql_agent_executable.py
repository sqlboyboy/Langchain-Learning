import os
import json
from decimal import Decimal
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
import mysql.connector

# 加载配置
load_dotenv()


# --- 1. 响应结构定义 ---
class SQLResponse(BaseModel):
    logic_explanation: str = Field(description="查询逻辑说明")
    sql_query: str = Field(description="最终生成的 SQL 语句")
    full_result: List[Dict[str, Any]] = Field(default_factory=list, description="结果数据列表")


# --- 2. 核心：解决 Decimal 序列化问题 ---
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


# --- 3. 数据库工具定义 ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE"),
    "auth_plugin": "mysql_native_password"
}


@tool
def get_db_schema() -> str:
    """获取数据库表结构 (DDL)。"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [t[0] if isinstance(t, (list, tuple)) else t for t in cursor.fetchall()]
        schemas = []
        for table in tables:
            cursor.execute(f"SHOW CREATE TABLE {table}")
            res = cursor.fetchone()
            if res:
                # res[1] 是真正的 CREATE TABLE 语句
                schemas.append(res[1] if len(res) > 1 else str(res))
        cursor.close()
        conn.close()
        return "\n\n".join(schemas)
    except Exception as e:
        return f"获取失败: {e}"


@tool
def execute_sql_query(query: str) -> str:
    """执行 SQL SELECT 语句并返回结果。"""
    print(f"\n🚀 [执行 SQL]: {query}")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        print(f"✅ [成功]: 获取到 {len(results)} 行数据")
        # 必须使用自定义 Encoder 处理 Decimal
        return json.dumps(results, cls=DecimalEncoder, ensure_ascii=False)
    except Exception as e:
        print(f"❌ [报错]: {e}")
        return f"查询报错: {e}"


# --- 4. 配置模型与 Agent ---
llm = ChatOpenAI(
    model="glm-4.5-air",
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    openai_api_base=os.getenv("ZHIPUAI_BASE_URL"),
    temperature=0.1,
    timeout=60
)

# 使用内存保存状态
memory = InMemorySaver()
# 创建 Agent
agent_executor = create_react_agent(
    llm,
    tools=[get_db_schema, execute_sql_query],
    checkpointer=memory
)


def run_sql_task(user_query: str):
    # 每次运行换一个 thread_id 确保状态干净
    config = {"configurable": {"thread_id": "sql_v_final_production"}}

    sys_msg = SystemMessage(content="""你是一位 SQL 专家。
    步骤：1.查结构 2.写SQL并执行 3.给出结论。
    注意：执行结果中如果有数字，请保留其含义，不要随意截断。""")

    print(f"\n" + "=" * 50)
    print(f"任务启动: {user_query}")
    print("=" * 50)

    final_output = ""
    history_messages = []

    # 1. 运行 Agent 并流式打印进度
    step_count = 0
    for event in agent_executor.stream({"messages": [sys_msg, HumanMessage(content=user_query)]}, config=config):
        step_count += 1
        if step_count > 12:  # 步数止损
            print("⚠️ [系统]: 步数过多，强制停止。")
            break

        for value in event.values():
            msg = value["messages"][-1]
            history_messages.append(msg)
            if msg.type == "ai" and msg.content:
                final_output = msg.content
                print(f"🤖 [模型回答]: {msg.content[:80]}...")

    # 2. 打印原始输出（双重保险）
    print("\n" + "-" * 20 + " 原始分析文本 " + "-" * 20)
    print(final_output)
    print("-" * 54)


# --- 5. 启动测试 ---
if __name__ == "__main__":
    task = "统计一下每公斤运费收入排名前三的寄件省份"
    try:
        res = run_sql_task(task)
        if res:
            print("\n" + " ✨ 最终结构化结果 ".center(50, "="))
            print(f"逻辑分析: {res.logic_explanation}")
            print(f"执行 SQL: {res.sql_query}")
            print(f"结果数据: {res.full_result}")
            print("=" * 50)
    except Exception as e:
        print(f"程序中断: {e}")

import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
import mysql.connector

load_dotenv()


# 1. 定义响应格式
class SQLResponse(BaseModel):
    logic_explanation: str = Field(description="查询逻辑说明")
    sql_query: str = Field(description="最终生成的 SQL 语句")
    is_verified: bool = Field(description="是否通过工具验证成功")


# 2. 数据库工具
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE"),
    "auth_plugin": "mysql_native_password"
}


@tool
def get_db_schema() -> str:
    """获取数据库所有表的 DDL。"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [t[0] if isinstance(t, (list, tuple)) else t for t in cursor.fetchall()]
        schemas = []
        for table in tables:
            cursor.execute(f"SHOW CREATE TABLE {table}")
            res = cursor.fetchone()
            if res: schemas.append(res[1])
        cursor.close()
        conn.close()
        return "\n\n".join(schemas)
    except Exception as e:
        return f"获取失败: {e}"


@tool
def run_sql_query(query: str) -> str:
    """执行 SQL 并返回结果。"""
    if not query.strip().lower().startswith("select"):
        return "错误：仅支持 SELECT。"
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return str(results[:5])
    except Exception as e:
        return f"SQL 报错: {e}"


# 3. 配置模型与 Agent
llm = ChatOpenAI(
    model="glm-4.5-air",
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    openai_api_base=os.getenv("ZHIPUAI_BASE_URL"),
    temperature=0.1
)

# 关键：将 SQLResponse 绑定为模型可选的输出方式
# 这样模型在完成任务后，会倾向于调用这个结构化输出
tools = [get_db_schema, run_sql_query]
memory = InMemorySaver()
agent_executor = create_react_agent(llm, tools=tools, checkpointer=memory)


def run_task(user_query: str):
    config = {"configurable": {"thread_id": "sql_session_v5"}}

    # 强制模型在最后必须输出指定的结构
    prompt = f"""你是一位 SQL 专家。
    1. 先查结构，再写 SQL，最后验证。
    2. 验证无误后，你必须使用最后一步的输出将结果格式化。
    格式要求：
    - logic_explanation: 你的分析说明。
    - sql_query: 验证过的 SQL。
    - is_verified: true。
    """

    messages = [SystemMessage(content=prompt), HumanMessage(content=user_query)]

    # 1. 正常执行 Agent
    result = agent_executor.invoke({"messages": messages}, config=config)
    final_text = result["messages"][-1].content

    # 2. 这里是关键：不再直接解析，而是利用 with_structured_output 重新格式化
    # 即使 final_text 是纯文字，这个方法也能强制 LLM 把纯文字转成 JSON
    structured_llm = llm.with_structured_output(SQLResponse)
    return structured_llm.invoke(f"根据以下信息提取结构化数据：\n{final_text}")


if __name__ == "__main__":
    query = "统计一下每公斤运费收入排名前三的寄件省份"
    try:
        res = run_task(query)
        print("\n--- 执行结果 ---")
        print(f"逻辑: {res.logic_explanation}")
        print(f"SQL: {res.sql_query}")
        print(f"验证: {res.is_verified}")
    except Exception as e:
        print(f"解析依然失败。建议打印 model 原始输出查验。错误: {e}")

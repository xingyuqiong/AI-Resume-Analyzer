from flask import Flask, request, jsonify
import pdfplumber
from io import BytesIO
import os
import json
from dotenv import load_dotenv
from zhipuai import ZhipuAI
from flask_cors import CORS


# ------------ 创建 Flask 应用 ------------
app = Flask(__name__)
CORS(app, resources={r"/resume/*": {"origins": "*"}})  # 允许所有来源访问

# ------------ 大模型API配置 ------------
# 加载 .env 里的配置
load_dotenv()

# 从环境变量里读 API Key 和模型名
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
GLM_MODEL = os.getenv("GLM_MODEL", "glm-4-flash")

if not ZHIPU_API_KEY:
    # 开发阶段直接抛错，提醒自己没有配置 key
    raise RuntimeError("缺少 ZHIPU_API_KEY，请在 .env 中配置")

# 初始化 Zhipu 客户端（后面所有请求都用它）
zhipu_client = ZhipuAI(api_key=ZHIPU_API_KEY)

# ---- 简单“内存数据库”，后面可以换成 Redis / 真数据库 ----
# key: resume_id, value: 解析后的简历结构化信息
RESUME_STORE = {}
NEXT_RESUME_ID = 1


def extract_text_from_pdf(file_storage):
    """
    使用 pdfplumber 从上传的 PDF 中提取纯文本
    :param file_storage: Flask 的 FileStorage 对象（request.files["file"]）
    :return: 清洗后的简历文本
    """
    # 读取文件字节内容
    file_bytes = file_storage.read()

    # 用 BytesIO 包一层，让 pdfplumber 可以当“文件”来打开
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        texts = []
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)

    # 合并多页文本
    text = "\n".join(texts)

    # 简单清洗：去掉空行、首尾空格
    lines = [
        line.strip()
        for line in text.splitlines()
        if line and line.strip()
    ]
    cleaned = "\n".join(lines)
    return cleaned


def extract_info_with_llm(raw_text: str) -> dict:
    """
    调用 GLM-4-Flash，从简历原文中抽取结构化关键信息。
    返回一个 Python dict，如果模型没有返回合法 JSON，就把原始字符串塞进 raw_output。
    """

    # 定义返回的 JSON 结构
    schema_hint = """
{
  "basic_info": {
    "name": "",
    "phone": "",
    "email": "",
    "address": ""
  },
  "education": [
    {
      "school": "",
      "degree": "",
      "major": "",
      "start_date": "",
      "end_date": ""
    }
  ],
  "work_experiences": [
    {
      "company": "",
      "title": "",
      "start_date": "",
      "end_date": "",
      "description": ""
    }
  ],
  "project_experiences": [
    {
      "project_name": "",
      "role": "",
      "start_date": "",
      "end_date": "",
      "description": ""
    }
  ],
  "campus_experiences": [
    {
      "organization": "",
      "title": "",
      "start_date": "",
      "end_date": "",
      "description": ""
    }
  ],
  "awards": [
    {
      "name": "",
      "level": "",
      "date": "",
      "description": ""
    }
  ],
  "skills": []
  "desired_positions": []
}
"""

    # 给模型的提示词：说明任务 + 要求只输出 JSON
    user_prompt = f"""
你现在是一个招聘系统里的“简历解析助手”。

我会给你一份中文简历的完整文本，请你：
1. 仔细阅读简历内容；
2. 按照下面这个 JSON 模板，从中抽取关键信息：
{schema_hint}
3. 所有字段没有就填 null、"" 或空数组；
4. **只输出 JSON 字符串，不要任何解释性文字**。

下面是简历全文（可能会比较长）：
--------------------
{raw_text}
--------------------
"""

    # 调用 Zhipu SDK（和文档里的例子是一致的）
    response = zhipu_client.chat.completions.create(
        model=GLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一个严谨的简历解析助手，只返回 JSON。"
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ],
        # 这里可以额外控制温度、max_tokens 等参数
        extra_body={"temperature": 0.2, "max_tokens": 1024},
    )

    # 模型返回的内容在这里
    content = response.choices[0].message.content

    # 尝试把它当 JSON 解析
    try:
        data = json.loads(content)
    except Exception:
        # 解析失败时，至少把原始输出保留起来，方便调试
        data = {"raw_output": content}

    return data


def match_resume_with_llm(parsed_resume: dict, job_description: str) -> dict:
    """
    调用 GLM，根据解析后的简历信息和岗位 JD，计算匹配度评分。
    """

    resume_json = json.dumps(parsed_resume, ensure_ascii=False, indent=2)

    schema_hint = """
{
  "overall_score": 0,
  "skill_match_score": 0,
  "experience_match_score": 0,
  "education_match_score": 0,
  "highlights": [],
  "risks": [],
  "reasoning": ""
}
"""

    user_prompt = f"""
你现在是一个资深 HR，请你根据下面的“简历信息”和“岗位 JD”进行匹配度分析。

【简历信息（JSON）】：
{resume_json}

【岗位 JD 文本】：
{job_description}

请你：
1. 从技能、工作经历、教育背景等角度进行匹配分析；
2. 给出 0-100 的综合匹配度评分 overall_score；
3. 额外给出 skill_match_score / experience_match_score / education_match_score 三个子评分；
4. 总结 2-5 条候选人的亮点（highlights）和 2-5 条潜在风险或不足（risks）；
5. 最后给出一段整体评语（reasoning）。

请严格按照下面这个 JSON 模板输出，不要任何多余文字：
{schema_hint}
"""

    response = zhipu_client.chat.completions.create(
        model=GLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一个严谨的候选人匹配度分析助手，只返回 JSON。"
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ],
        extra_body={"temperature": 0.2, "max_tokens": 1024},
    )

    content = response.choices[0].message.content

    try:
        data = json.loads(content)
    except Exception:
        data = {"raw_output": content}

    return data


# 连接性检查接口：GET /resume/test
@app.route("/resume/test", methods=["GET"])
def test():
    return jsonify({"status": "ok"})


# 简历上传接口：POST /resume/upload
@app.route("/resume/upload", methods=["POST"])
def upload_resume():
    """
    简历上传与解析接口
    - 接收 form-data 中的 file 字段（PDF 文件）
    - 解析出文本，先不调用大模型，只做解析 + 存储
    - 返回 resume_id 和部分文本预览
    """
    global NEXT_RESUME_ID

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "缺少文件字段 file"}), 400

    # 只允许 PDF
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "目前只支持 PDF 文件"}), 400

    try:
        raw_text = extract_text_from_pdf(file)
    except Exception as e:
        return jsonify({"error": f"PDF 解析失败: {e}"}), 500

    # 调用 GLM 抽取关键信息
    try:
        parsed_info = extract_info_with_llm(raw_text)
    except Exception as e:
        # 模型调用失败时，不影响基础功能，parsed 置为 None
        parsed_info = None

    # 生成一个 resume_id
    resume_id = str(NEXT_RESUME_ID)
    NEXT_RESUME_ID += 1

    # 保存到“内存数据库”里
    RESUME_STORE[resume_id] = {
        "filename": file.filename,
        "raw_text": raw_text,
        "parsed": parsed_info  # 现在这里就存了 AI 提取结果
    }

    # # 只返回前 500 个字符做预览，避免太长
    # preview = raw_text[:500]

    return jsonify({
        "message": "简历上传解析成功，并已完成 AI 关键信息抽取",
        "resume_id": resume_id,
        "filename": file.filename,
        # "text_preview": preview,
        "parsed": parsed_info,  # 把解析结果直接返回给前端/调用方
    })


# 简历与职位匹配接口：POST /resume/match
@app.route("/resume/match", methods=["POST"])
def match_resume():
    """
    简历信息 和 岗位 JD 匹配度 计算接口
    """
    data = request.get_json(silent=True) or {}

    resume_id = data.get("resume_id")
    job_description = data.get("job_description")

    if not resume_id or not job_description:
        return jsonify({"error": "resume_id 和 job_description 都是必填"}), 400

    resume = RESUME_STORE.get(resume_id)
    if not resume:
        return jsonify({"error": f"找不到简历 {resume_id}，请先调用 /resume/upload 上传简历"}), 404

    # 如果之前解析失败，这里补一次（防御性写法）
    if not resume.get("parsed"):
        resume["parsed"] = extract_info_with_llm(resume["raw_text"])

    # 调用 GLM 进行匹配度分析
    match_result = match_resume_with_llm(resume["parsed"], job_description)

    # 也可以选择把匹配结果缓存回 RESUME_STORE
    resume["last_match"] = {
        "job_description": job_description,
        "match_result": match_result,
    }

    return jsonify({
        "resume_id": resume_id,
        "job_description": job_description,
        "match_result": match_result,       # 评分和分析
        # "parsed_resume": resume["parsed"],  # 解析后的简历信息
    })


if __name__ == "__main__":
    # 阿里云 FC 要求 9000；本地：默认 8000
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
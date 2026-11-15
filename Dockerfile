# 用 Docker 執行上傳照片版（app.py）
# 打包：docker build -t ai-qa-assistant .
# 執行：docker run --rm -p 8501:8501 ai-qa-assistant
# 要用 GPT 確認的話，把本機的金鑰檔掛進去（金鑰不會被打包進映像檔）：
#   docker run --rm -p 8501:8501 -v "$PWD/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro" ai-qa-assistant

FROM python:3.12-slim

# OpenCV 需要的系統套件，跟 Streamlit Cloud 用的 packages.txt 一樣
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先只複製 requirements.txt 安裝套件：程式改了但套件沒變時，這一層可以直接用上次的結果，不用重裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再複製程式、設定、SOP、模型（要放哪些檔案寫在 .dockerignore）
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

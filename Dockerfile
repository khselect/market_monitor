FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 MM_HOME=/app MM_DATA=/data TZ=Asia/Seoul
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
# MM_BASE_PATH: 서브디렉토리 게시용(예: market → https://sielain.com/market). 서브도메인이면 비워둠
CMD ["sh", "-c", "streamlit run app/dashboard.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --server.baseUrlPath=${MM_BASE_PATH:-}"]

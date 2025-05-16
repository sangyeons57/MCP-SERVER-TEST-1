FROM python:3.10-slim

WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY calculator_mcp.py .

# uvicorn이 사용할 8000 포트 노출
EXPOSE 8000

# 컨테이너 실행 시 Python 스크립트를 직접 실행
CMD ["python", "calculator_mcp.py"]

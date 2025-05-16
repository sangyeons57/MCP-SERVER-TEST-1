FROM python:3.11-slim

WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
# 필요한 모든 .py 파일을 복사합니다.
COPY main_mcp_server.py .
COPY mcp_tools/ ./mcp_tools/
# 만약 stats_tools.py 등 다른 모듈이 추가되면 여기에 COPY 구문을 추가합니다.

# uvicorn(mcp.run 내부에서 사용)이 사용할 8000 포트 노출
EXPOSE 8000

# 컨테이너 실행 시 메인 서버 스크립트를 실행
CMD ["python", "main_mcp_server.py"]

# MCP 계산기 서버

이 프로젝트는 Model Context Protocol(MCP)을 사용하는 간단한 계산기 서버를 구현합니다. Cursor IDE와 연동하여 기본적인 수학 계산 기능을 제공합니다.

## 기능

- 기본 산술 연산:
  - 더하기 (`add`)
  - 빼기 (`subtract`)
  - 곱하기 (`multiply`)
  - 나누기 (`divide`)
  - 거듭제곱 (`power`)
- Docker 컨테이너를 통한 간편한 배포
- Cursor IDE와 연동

## 설치 및 실행 방법

### 사전 요구사항
- Docker 및 Docker Compose 설치
- Cursor IDE

### 로컬에서 실행하기

1. 저장소 클론:
   ```bash
   git clone https://github.com/사용자명/mcp-calculator.git
   cd mcp-calculator
   ```

2. Docker 컨테이너 빌드 및 실행:
   ```bash
   docker-compose up --build -d
   ```

3. 확인:
   ```bash
   curl -v http://localhost:8001/sse
   ```
   정상적으로 동작하면 `200 OK`와 함께 SSE 스트림이 시작됩니다.

## Cursor IDE와 연결하기

1. Cursor IDE를 실행합니다.
2. 명령 팔레트(Windows/Linux: `Ctrl+Shift+P`, Mac: `Cmd+Shift+P`)에서 `Cursor Settings`를 검색합니다.
3. 설정에서 `MCP` 섹션으로 이동합니다.
4. `Add Custom MCP Server`를 클릭합니다.
5. MCP 서버 URL로 `http://localhost:8001/sse`를 입력합니다.
6. 이름을 `CalculatorMCP`로 지정합니다.
7. Cursor를 재시작합니다.

## 사용 예시

Cursor에서 다음과 같이 사용할 수 있습니다:

```
@CalculatorMCP add 123 and 456
@CalculatorMCP multiply 78 by 9
@CalculatorMCP 5 to the power of 3
```

## GitHub에서 프로젝트 포크하기

1. GitHub 저장소 페이지에서 "Fork" 버튼을 클릭합니다.
2. 포크된 저장소를 로컬에 클론합니다.
3. 필요한 경우 코드를 수정합니다.
4. 변경사항을 커밋하고 푸시합니다.

## 라이센스

MIT

## 기여하기

Pull Request는 언제든지 환영합니다!
from mcp.server.fastmcp import FastMCP

# MCP 서버 인스턴스 생성
mcp = FastMCP("CalculatorMCP")

@mcp.tool()
def add(a: float, b: float) -> float:
    """두 숫자를 더합니다
    
    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자
        
    Returns:
        두 숫자의 합
    """
    return a + b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """두 숫자를 곱합니다
    
    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자
        
    Returns:
        두 숫자의 곱
    """
    return a * b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """첫 번째 숫자에서 두 번째 숫자를 뺍니다
    
    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자
        
    Returns:
        두 숫자의 차
    """
    return a - b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """첫 번째 숫자를 두 번째 숫자로 나눕니다
    
    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자 (0이 아니어야 함)
        
    Returns:
        두 숫자의 몫
        
    Raises:
        ZeroDivisionError: b가 0일 경우
    """
    if b == 0:
        raise ValueError("0으로 나눌 수 없습니다.")
    return a / b

@mcp.tool()
def power(a: float, b: float) -> float:
    """첫 번째 숫자를 두 번째 숫자의 거듭제곱으로 계산합니다
    
    Args:
        a: 밑수
        b: 지수
        
    Returns:
        a의 b승
    """
    return a ** b

@mcp.resource("calculator://info")
def calculator_info() -> str:
    """계산기 정보를 제공합니다"""
    return "이 계산기는 기본적인 산술 연산(더하기, 빼기, 곱하기, 나누기, 거듭제곱)을 제공합니다."

# 서버를 실행하는 코드
if __name__ == "__main__":
    print("MCP Calculator 서버 (mcp.run, transport='sse' 사용) 시작 중...")
    # host와 port 인자를 제거하고 transport만 지정합니다.
    mcp.run(transport="sse")
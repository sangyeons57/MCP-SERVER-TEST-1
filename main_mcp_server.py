from mcp.server.fastmcp import FastMCP

# Import tool-providing functions from other modules
from arithmetic_tools import get_arithmetic_tools
from linalg_tools import get_linalg_tools

# Create the main MCP server instance
mcp = FastMCP("CalculatorMCP") # Server name remains the same for Cursor

# Register tools from arithmetic_tools.py
for tool_func in get_arithmetic_tools():
    mcp.tool()(tool_func)

# Register tools from linalg_tools.py
for tool_func in get_linalg_tools():
    mcp.tool()(tool_func)

# Define and register the info resource directly in the main server file
@mcp.resource("calculator://info")
def calculator_info() -> str:
    """Provides information about this calculator server and its capabilities."""
    # This description should be a consolidated view of all registered tools
    # For simplicity, we'll manually craft it here for now.
    # A more dynamic approach might inspect registered tools if feasible.
    return (
        "This calculator provides: \n"
        "- Basic Arithmetic: add, subtract, multiply, divide, modulo, power, sqrt, absolute.\n"
        "- Advanced Math: log (custom base), ln (natural log), log10 (common log), sin_degrees, cos_degrees, tan_degrees, factorial.\n"
        "- Statistics: average.\n"
        "- Linear Algebra: matrix_add, matrix_subtract, matrix_scalar_multiply, matrix_multiply, transpose."
    )

# Code to run the server
if __name__ == "__main__":
    print(f"Starting {mcp.name} MCP Server with modular tools...")
    print(f"Registered tools from arithmetic_tools: {len(get_arithmetic_tools())}")
    print(f"Registered tools from linalg_tools: {len(get_linalg_tools())}")
    print(f"Resource 'calculator://info' is available.")
    mcp.run(transport="sse") 
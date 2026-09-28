from fastmcp import FastMCP

mcp = FastMCP("dummy")


@mcp.tool()
def ping() -> str:
    """Respond with pong."""
    return "pong"


if __name__ == "__main__":
    mcp.run()

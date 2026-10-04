# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp==2.3.0"]
# ///
"""配套天气MCP服务；依据官方天气教程重新实现，数据来自NWS。"""
import asyncio
import json
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from mcp.server import MCPServer

mcp = MCPServer("book-weather", instructions="美国天气预报与警报。按返回的当地时间解释日期，温度带单位；调用失败不能编造天气。")
BASE = "https://api.weather.gov"

async def fetch(url: str) -> dict:
    def read():
        req = Request(url, headers={"User-Agent": "ai-terms-book-weather/1.0", "Accept": "application/geo+json"})
        try:
            with urlopen(req, timeout=25) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as error:
            raise RuntimeError(f"NWS请求失败：{error}。未取得天气结果，请稍后重试。") from error
    return await asyncio.to_thread(read)

@mcp.tool()
async def get_forecast(latitude: float = 40.7128, longitude: float = -74.0060) -> dict:
    """查询美国地点预报。默认纽约市；其他美国地点传经纬度。使用每段startTime的当地日期判断明天。"""
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("经纬度超出范围")
    points = await fetch(f"{BASE}/points/{latitude},{longitude}")
    url = points["properties"]["forecast"]
    forecast = await fetch(url)
    properties = forecast["properties"]
    fields = ("name", "startTime", "endTime", "isDaytime", "temperature", "temperatureUnit", "windSpeed", "windDirection", "shortForecast", "detailedForecast")
    return {"source": "美国国家气象局NWS", "source_url": url,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "updated": properties.get("updated"),
            "periods": [{key: item.get(key) for key in fields} for item in properties["periods"][:8]]}

@mcp.tool()
async def get_alerts(state: str = "NY") -> dict:
    """查询美国州的当前警报。默认纽约州NY；州警报不等于纽约市一定有警报。"""
    state = state.upper().strip()
    if len(state) != 2 or not state.isascii() or not state.isalpha():
        raise ValueError("请输入美国州的两字母代码，例如NY")
    url = f"{BASE}/alerts/active?area={state}"
    data = await fetch(url)
    fields = ("event", "areaDesc", "severity", "effective", "expires", "headline", "description", "instruction")
    return {"source": "美国国家气象局NWS", "source_url": url, "state": state,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "alerts": [{key: item["properties"].get(key) for key in fields} for item in data.get("features", [])],
            "note": "没有警报与请求失败不同；本返回列出当前接口提供的警报。"}

if __name__ == "__main__":
    mcp.run(transport="stdio")

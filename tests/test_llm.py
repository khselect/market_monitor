"""LLM 브리핑: 실제 anthropic SDK → 로컬 모의 서버로 요청 형식·응답 파싱·pause_turn 재개 검증 (네트워크/비용 0)."""
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

os.environ.setdefault("MM_DATA", tempfile.mkdtemp())
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mm.llm import briefing  # noqa: E402

CALLS = []


def _msg(content, stop="end_turn"):
    return {"id": "msg_x", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 50}}


SEARCH_BLOCKS = [
    {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "nasdaq"}},
    {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
     "content": [{"type": "web_search_result", "title": "Stocks rally", "url": "https://ex.com/a",
                  "encrypted_content": "x", "page_age": None}]},
]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        CALLS.append({"body": body, "beta": self.headers.get("anthropic-beta")})
        if "format" in body.get("output_config", {}):
            out = {"headline": "h", "market_brief": ["a"], "korea_brief": [], "event_watch": [],
                   "scenarios": [{"id": "base", "narrative": "n", "triggers": ["t"], "portfolio_note": "p"}],
                   "stock_news": [{"title": "Stocks rally", "summary": "s", "url": "https://ex.com/a"}],
                   "ai_trends": [], "risks": []}
            resp = _msg([{"type": "text", "text": json.dumps(out)}])
        elif len([c for c in CALLS if "tools" in c["body"]]) == 1:
            resp = _msg(SEARCH_BLOCKS, stop="pause_turn")          # 첫 리서치 호출은 일시정지
        else:
            resp = _msg([{"type": "text", "text": "메모: 증시 상승 (https://ex.com/a)"}])
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_briefing_roundtrip_with_sdk(monkeypatch):
    import anthropic
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    client = anthropic.Anthropic(api_key="test", base_url=f"http://127.0.0.1:{srv.server_port}", max_retries=0)
    facts = {"asof": "2026-11-05", "panels": {}, "scenarios": [{"id": "base", "label": "x", "prob_next_month": 0.7}]}
    monkeypatch.setattr(briefing, "build_facts", lambda: facts)
    monkeypatch.setattr(briefing, "yahoo_news", lambda kind: [])
    monkeypatch.setattr(briefing, "alphavantage_news", lambda: [])
    out = briefing.generate(weekly=False, client=client)
    srv.shutdown()

    assert out["_meta"]["generator"] == "claude"
    assert out["stock_news"][0]["url"] == "https://ex.com/a"
    assert {"title": "Stocks rally", "url": "https://ex.com/a"} in out["_meta"]["sources"]
    research = [c for c in CALLS if "tools" in c["body"]]
    assert len(research) == 3                                        # market(일시정지→재개) + ai
    assert research[1]["body"]["messages"][-1]["role"] == "assistant"  # pause_turn 재개 시 assistant 턴 첨부
    for c in CALLS:
        assert c["body"]["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in c["beta"]
        assert c["body"]["thinking"] == {"type": "adaptive"}
    synth = [c for c in CALLS if "format" in c["body"].get("output_config", {})][0]["body"]
    assert synth["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert "FACTS" in synth["messages"][0]["content"]


def test_parse_yf_news_both_formats():
    old = [{"title": "A", "link": "https://x/a", "publisher": "Reuters", "providerPublishTime": 1762400000}]
    new = [{"id": "1", "content": {"title": "B", "summary": "s", "pubDate": "2026-11-06T10:00:00Z",
                                   "canonicalUrl": {"url": "https://x/b"}, "provider": {"displayName": "Yahoo"}}},
           {"id": "2", "content": {"title": "no url"}}]
    out = briefing.parse_yf_news(old) + briefing.parse_yf_news(new)
    assert [o["url"] for o in out] == ["https://x/a", "https://x/b"] and out[1]["source"] == "Yahoo"


def test_template_uses_feeds_and_records_reason(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    facts = {"asof": "2026-11-05", "panels": {}, "scenarios": []}
    monkeypatch.setattr(briefing, "build_facts", lambda: facts)
    feed = [{"title": "Fed holds", "url": "https://x/f", "source": "Reuters", "time": "2026-11-06", "summary": ""}]
    monkeypatch.setattr(briefing, "yahoo_news", lambda kind: feed)
    monkeypatch.setattr(briefing, "alphavantage_news", lambda: [])
    out = briefing.generate()
    assert out["_meta"]["generator"] == "template" and "ANTHROPIC_API_KEY" in out["_meta"]["diag"]["reason"]
    assert out["stock_news"][0]["url"] == "https://x/f" and out["ai_trends"][0]["title"] == "Fed holds"


def test_web_search_error_object_is_recorded():
    """웹 검색 비활성 등 서버 도구 오류는 HTTP 200 + 오류 객체 → 진단에 기록돼야 함(이전엔 무시됨)."""
    from types import SimpleNamespace as N
    resp = N(stop_reason="end_turn", content=[
        N(type="web_search_tool_result", content=N(type="web_search_tool_result_error", error_code="unavailable")),
        N(type="text", text="검색 불가")])
    client = N(beta=N(messages=N(create=lambda **kw: resp)))
    diag = {}
    r = briefing._research(client, "market", 1, 1, diag)
    assert r["sources"] == [] and diag["research"][0]["errors"] == ["unavailable"]

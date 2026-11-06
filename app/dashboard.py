"""Market Monitor — 진입점 (멀티페이지).

  /            개요: 9개 패널 한 화면 (1920×1080)
  /indicators  주요지표   /commodities 원자재   /treasury 미 국채
  /sectors     산업별     /forecast    예상      /briefing AI 브리핑
  /portfolio   10년 포트폴리오   /goals 목표 판정   /model 모델 상태

개요 패널 제목 옆 ⤢ 를 누르면 해당 상세 페이지가 새 탭(전체화면)으로 열림.
서브디렉토리 게시(MM_BASE_PATH) 시에도 링크는 baseUrlPath 를 반영한 절대 경로로 생성.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import inject_css  # noqa: E402
from views import details as d  # noqa: E402

st.set_page_config(page_title="Market Monitor", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
inject_css()

pages = [
    st.Page("views/overview.py", title="개요", default=True),
    st.Page(d.page_indicators, title="주요지표", url_path="indicators"),
    st.Page(d.page_commodities, title="원자재", url_path="commodities"),
    st.Page(d.page_treasury, title="미 국채", url_path="treasury"),
    st.Page(d.page_sectors, title="산업별", url_path="sectors"),
    st.Page(d.page_forecast, title="예상", url_path="forecast"),
    st.Page(d.page_briefing, title="AI 브리핑", url_path="briefing"),
    st.Page(d.page_portfolio, title="10년 포트폴리오", url_path="portfolio"),
    st.Page(d.page_goals, title="목표 판정", url_path="goals"),
    st.Page(d.page_model, title="모델 상태", url_path="model"),
]
st.navigation(pages, position="hidden").run()

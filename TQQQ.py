import math
import time
import random
import requests
from datetime import date, datetime

import pandas as pd
import yfinance as yf
import streamlit as st


# ----------------- Network / Retry -----------------
def make_session() -> requests.Session:
    s = requests.Session()
    # 常见 UA，某些情况下能更稳一点
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123 Safari/537.36"
        )
    })
    return s


SESSION = make_session()


def is_rate_limited_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("ratelimit" in msg) or ("too many requests" in msg) or ("429" in msg)


def with_retry(fn, tries: int = 5, base_sleep: float = 1.2, jitter: float = 0.6):
    """
    针对 Yahoo 限流/网络抖动：指数退避重试
    """
    last_err = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if is_rate_limited_error(e):
                sleep_s = base_sleep * (2 ** i) + random.uniform(0, jitter)
                time.sleep(sleep_s)
                continue
            # 其他错误不盲目重试
            raise
    raise last_err


# ----------------- Helpers -----------------
def _to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _dte(expiry: str, today: date | None = None) -> int:
    today = today or date.today()
    return max((_to_date(expiry) - today).days, 0)


def option_mark_mid_or_last(bid, ask, last):
    """优先 mid；非交易时段/无报价时兜底 lastPrice"""
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2.0
    if pd.notna(last) and last and last > 0:
        return float(last)
    return math.nan


def put_sell_price_conservative_or_fallback(row: pd.Series) -> float:
    """
    卖 put 的更保守成交价：
      1) 优先 bid（更贴近你能卖到的价格）
      2) bid 不可用则回退 mid/last
    """
    bid = row.get("bid")
    if pd.notna(bid) and bid > 0:
        return float(bid)
    return option_mark_mid_or_last(row.get("bid"), row.get("ask"), row.get("lastPrice"))


@st.cache_data(ttl=300, show_spinner=False)  # 5分钟缓存
def get_underlying_price(symbol: str) -> float:
    """标的当前价/最后价：多重兜底（带缓存 + 重试）"""
    t = yf.Ticker(symbol, session=SESSION)

    # 1) fast_info last_price（常见可用）
    try:
        fi = getattr(t, "fast_info", None)
        if fi and fi.get("last_price"):
            return float(fi["last_price"])
    except Exception:
        pass

    # 2) 1m 最后一根
    try:
        h = with_retry(lambda: t.history(period="1d", interval="1m"))
        if not h.empty and pd.notna(h["Close"].iloc[-1]):
            return float(h["Close"].iloc[-1])
    except Exception:
        pass

    # 3) 5d 最后一根日线 close
    try:
        h = with_retry(lambda: t.history(period="5d", interval="1d"))
        if not h.empty and pd.notna(h["Close"].iloc[-1]):
            return float(h["Close"].iloc[-1])
    except Exception:
        pass

    return float("nan")


def fmt_pct(x):
    """把 0.2345 显示成 23.45%"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x * 100:.2f}%"


# ----------------- Yahoo data (cached) -----------------
@st.cache_data(ttl=600, show_spinner=False)  # 10分钟缓存
def get_expirations(symbol: str) -> list[str]:
    t = yf.Ticker(symbol, session=SESSION)
    return with_retry(lambda: list(t.options) or [])


@st.cache_data(ttl=600, show_spinner=False)  # 10分钟缓存
def get_put_chain(symbol: str, exp: str) -> pd.DataFrame:
    t = yf.Ticker(symbol, session=SESSION)
    chain = with_retry(lambda: t.option_chain(exp))
    return chain.puts.copy()


# ----------------- Core Scan -----------------
def scan_puts_ws_margin_rule(
    symbol: str,
    min_dte: int,
    max_dte: int,
    underlying_px: float,
    otm_pct: float | None,
    strike_min: float | None,
    strike_max: float | None,
    top_n_per_expiry: int,
    min_annualized: float | None,
) -> tuple[pd.DataFrame, dict]:
    """
    Wealthsimple 保证金口径：
      margin_per_share = strike - premium
      annualized = (premium_$ / margin_$) * (365 / dte)

    premium：优先 bid（保守卖出价），否则回退 mid/last。

    支持：
      - OTM%：strike <= 现价*(1-OTM)
      - 按到期日分组 Top N
    """

    expirations = get_expirations(symbol)

    meta = {
        "computed_strike_cap_from_otm": None,
        "underlying_px": underlying_px,
        "total_rows_before_grouping": 0,
        "total_rows_after_grouping": 0,
        "expirations_in_range": 0,
    }

    # 根据 OTM% 计算 strike cap（上限）
    strike_cap = None
    if otm_pct is not None and pd.notna(underlying_px) and underlying_px > 0:
        strike_cap = underlying_px * (1.0 - otm_pct)
        meta["computed_strike_cap_from_otm"] = strike_cap

    rows = []
    expirations_in_range = []

    for exp in expirations:
        dte = _dte(exp)
        if dte < min_dte or dte > max_dte:
            continue
        expirations_in_range.append(exp)

    meta["expirations_in_range"] = len(expirations_in_range)

    for exp in expirations_in_range:
        dte = _dte(exp)

        puts = get_put_chain(symbol, exp)
        if puts.empty:
            continue

        # strike filters (manual)
        if strike_min is not None:
            puts = puts.loc[puts["strike"] >= float(strike_min)]
        if strike_max is not None:
            puts = puts.loc[puts["strike"] <= float(strike_max)]

        # strike filter (OTM cap)
        if strike_cap is not None:
            puts = puts.loc[puts["strike"] <= float(strike_cap)]

        if puts.empty:
            continue

        for _, r in puts.iterrows():
            strike = float(r["strike"])

            premium_per_share = put_sell_price_conservative_or_fallback(r)
            if not (pd.notna(premium_per_share) and premium_per_share > 0):
                continue

            margin_per_share = strike - premium_per_share
            if margin_per_share <= 0:
                continue

            premium_dollars = premium_per_share * 100
            margin_dollars = margin_per_share * 100

            ann = (premium_dollars / margin_dollars) * (365 / dte)

            if min_annualized is not None and ann < float(min_annualized):
                continue

            rows.append({
                "symbol": symbol,
                "expiration": exp,
                "dte": dte,
                "strike": strike,

                "bid": r.get("bid"),
                "ask": r.get("ask"),
                "lastPrice": r.get("lastPrice"),

                "premium_used_per_share": premium_per_share,
                "premium_$": premium_dollars,

                "margin_per_share_(K-premium)": margin_per_share,
                "margin_$": margin_dollars,

                "annualized": ann,
                "volume": r.get("volume"),
                "openInterest": r.get("openInterest"),
                "impliedVolatility": r.get("impliedVolatility"),
                "inTheMoney": r.get("inTheMoney"),
            })

    out = pd.DataFrame(rows)
    meta["total_rows_before_grouping"] = len(out)

    if out.empty:
        return out, meta

    # Top N per expiry
    top_n = max(int(top_n_per_expiry), 1)
    out = out.sort_values(["expiration", "annualized", "premium_$"], ascending=[True, False, False])
    out = out.groupby("expiration", as_index=False, group_keys=False).head(top_n)

    out = out.sort_values(["annualized", "premium_$"], ascending=[False, False]).reset_index(drop=True)
    meta["total_rows_after_grouping"] = len(out)

    return out, meta


# ----------------- UI -----------------
st.set_page_config(page_title="TQQQ Put Annualized Scanner (WS Margin Rule)", layout="wide")
st.title("TQQQ 卖 Put 年化收益率扫描器（按 Wealthsimple 保证金口径）")

symbol = st.sidebar.text_input("Symbol", value="TQQQ").upper()

st.sidebar.markdown("### 到期日区间（DTE）")
min_dte = st.sidebar.slider("最小 DTE", 1, 180, 7)
max_dte = st.sidebar.slider("最大 DTE", 1, 365, 30)

st.sidebar.markdown("### OTM 过滤（按现价自动算 strike 上限）")
use_otm = st.sidebar.checkbox("启用 OTM% 过滤", value=True)
otm_pct = None
if use_otm:
    otm_pct = st.sidebar.slider("OTM%", 0.0, 0.6, 0.10, 0.01)

st.sidebar.markdown("### Strike 手动过滤（可选）")
strike_min_in = st.sidebar.number_input("Strike Min（0 表示不限制）", value=0.0, step=1.0)
strike_max_in = st.sidebar.number_input("Strike Max（0 表示不限制）", value=0.0, step=1.0)

st.sidebar.markdown("### 每个到期日 Top N")
top_n = st.sidebar.slider("Top N per Expiration", 1, 50, 10)

st.sidebar.markdown("### 年化过滤（可选）")
min_ann_in = st.sidebar.number_input("Min Annualized（例如 0.20=20%）", value=0.0, step=0.01)

run = st.sidebar.button("运行扫描")

# show underlying price (cached)
underlying_px = get_underlying_price(symbol)

c1, c2 = st.columns([1, 3])
with c1:
    st.metric(label=f"{symbol} 当前/最后价格", value=("N/A" if math.isnan(underlying_px) else f"{underlying_px:.2f}"))
with c2:
    st.caption(
        "卖 put 定价：优先用 bid（更保守）；若 bid 不可用则回退 mid/lastPrice。"
        " 保证金口径（Wealthsimple）：margin = Strike - premium。"
        "（已加入缓存 + 限流重试，Streamlit Cloud 更稳定）"
    )

if run:
    strike_min = None if strike_min_in <= 0 else float(strike_min_in)
    strike_max = None if strike_max_in <= 0 else float(strike_max_in)
    min_ann = None if min_ann_in <= 0 else float(min_ann_in)

    try:
        df, meta = scan_puts_ws_margin_rule(
            symbol=symbol,
            min_dte=min_dte,
            max_dte=max_dte,
            underlying_px=underlying_px,
            otm_pct=otm_pct,
            strike_min=strike_min,
            strike_max=strike_max,
            top_n_per_expiry=top_n,
            min_annualized=min_ann,
        )
    except Exception as e:
        if is_rate_limited_error(e):
            st.error(
                "Yahoo Finance 当前对请求限流（Rate Limit）。\n\n"
                "建议：\n"
                "• 等 1–5 分钟再试\n"
                "• 缩小 DTE 范围 / 提高 OTM% / 降低 Top N\n"
                "• 或换 Railway/Render 部署（独立 IP 更不容易被限流）"
            )
            st.stop()
        else:
            st.exception(e)
            st.stop()

    if otm_pct is not None and meta.get("computed_strike_cap_from_otm") is not None:
        st.info(
            f"OTM% 过滤启用：OTM={otm_pct*100:.0f}% → strike ≤ 现价×(1-OTM) = {meta['computed_strike_cap_from_otm']:.2f}"
        )

    st.caption(
        f"扫描到期日（DTE区间内）: {meta.get('expirations_in_range', 0)} 个；"
        f"分组前 {meta['total_rows_before_grouping']} 条；"
        f"按到期日 Top {top_n} 后 {meta['total_rows_after_grouping']} 条。"
    )

    if df.empty:
        st.warning("没有扫描到结果：可能是 DTE/OTM/strike 过滤太严格，或 Yahoo 数据暂时缺失。")
    else:
        show = df.copy()
        show["annualized(%)"] = show["annualized"].apply(fmt_pct)

        cols_order = [
            "expiration", "dte", "strike",
            "annualized(%)", "premium_$", "margin_$",
            "premium_used_per_share", "margin_per_share_(K-premium)",
            "bid", "ask", "lastPrice",
            "volume", "openInterest", "impliedVolatility", "inTheMoney",
        ]
        cols_order = [c for c in cols_order if c in show.columns]
        show = show[cols_order]

        st.dataframe(show, use_container_width=True)

        st.download_button(
            "下载 CSV（含 annualized 原始小数）",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{symbol}_put_annualized_scan.csv",
            mime="text/csv",
        )
else:
    st.caption("设置参数后点击左侧「运行扫描」。")

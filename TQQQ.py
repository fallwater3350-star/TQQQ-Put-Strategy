import math
from datetime import date, datetime
import pandas as pd
import yfinance as yf
import streamlit as st


# ---------- helpers ----------
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
    卖 put 的更保守成交价：优先用 bid（你更可能按 bid 成交），
    若 bid 不可用则回退 mid/last。
    """
    bid = row.get("bid")
    if pd.notna(bid) and bid > 0:
        return float(bid)
    return option_mark_mid_or_last(row.get("bid"), row.get("ask"), row.get("lastPrice"))

def get_underlying_price(t: yf.Ticker) -> float:
    """标的当前价/最后价：多重兜底"""
    try:
        fi = getattr(t, "fast_info", None)
        if fi and fi.get("last_price"):
            return float(fi["last_price"])
    except Exception:
        pass

    try:
        h = t.history(period="1d", interval="1m")
        if not h.empty and pd.notna(h["Close"].iloc[-1]):
            return float(h["Close"].iloc[-1])
    except Exception:
        pass

    try:
        h = t.history(period="5d", interval="1d")
        if not h.empty and pd.notna(h["Close"].iloc[-1]):
            return float(h["Close"].iloc[-1])
    except Exception:
        pass

    return float("nan")


def scan_cash_or_margin_secured_puts(
    symbol: str,
    min_dte: int,
    max_dte: int,
    strike_min: float | None = None,
    strike_max: float | None = None,
    min_annualized: float | None = None,
) -> pd.DataFrame:
    """
    扫描指定 DTE 区间内的 put，按 Wealthsimple 保证金口径计算年化：
      margin_per_share = strike - premium
      annualized = (premium / margin) * (365 / dte)

    premium 使用：优先 bid（保守卖出价），否则回退 mid/last。
    """
    t = yf.Ticker(symbol)
    expirations = list(t.options) or []
    rows = []

    for exp in expirations:
        dte = _dte(exp)
        if dte < min_dte or dte > max_dte:
            continue

        chain = t.option_chain(exp)
        puts = chain.puts.copy()
        if puts.empty:
            continue

        # strike 过滤
        if strike_min is not None:
            puts = puts.loc[puts["strike"] >= float(strike_min)]
        if strike_max is not None:
            puts = puts.loc[puts["strike"] <= float(strike_max)]
        if puts.empty:
            continue

        for _, r in puts.iterrows():
            strike = float(r["strike"])
            premium_per_share = put_sell_price_conservative_or_fallback(r)

            if not (pd.notna(premium_per_share) and premium_per_share > 0):
                continue

            # Wealthsimple margin per share (你提供的规则)
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

                # raw quotes
                "bid": r.get("bid"),
                "ask": r.get("ask"),
                "lastPrice": r.get("lastPrice"),

                # used premium
                "premium_used_per_share": premium_per_share,
                "premium_$": premium_dollars,

                # WS margin rule
                "margin_per_share_(K-premium)": margin_per_share,
                "margin_$": margin_dollars,

                "annualized": ann,
                "inTheMoney": r.get("inTheMoney"),
                "openInterest": r.get("openInterest"),
                "volume": r.get("volume"),
                "impliedVolatility": r.get("impliedVolatility"),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["annualized", "premium_$"], ascending=[False, False]).reset_index(drop=True)
    return out


# ---------- UI ----------
st.set_page_config(page_title="TQQQ Put Annualized Scanner (Wealthsimple Margin Rule)", layout="wide")
st.title("TQQQ 卖 Put 年化收益率扫描器（按 Wealthsimple 保证金口径）")

symbol = st.sidebar.text_input("Symbol", value="TQQQ").upper()
min_dte = st.sidebar.slider("最小 DTE", 1, 180, 7)
max_dte = st.sidebar.slider("最大 DTE", 1, 365, 30)

st.sidebar.markdown("### Strike 过滤（可选）")
strike_min = st.sidebar.number_input("Strike Min", value=0.0, step=1.0)
strike_max = st.sidebar.number_input("Strike Max（0 表示不限制）", value=0.0, step=1.0)

st.sidebar.markdown("### 年化过滤（可选）")
min_ann = st.sidebar.number_input("Min Annualized（例如 0.20=20%）", value=0.0, step=0.01)

run = st.sidebar.button("运行扫描")

t = yf.Ticker(symbol)
px = get_underlying_price(t)
st.metric(label=f"{symbol} 当前/最后价格", value=("N/A" if math.isnan(px) else f"{px:.2f}"))

st.caption(
    "定价规则：卖 put 优先用 bid（更保守）；若 bid 不可用则回退 mid/lastPrice。"
    "年化按你给的 Wealthsimple 保证金口径：margin = K - premium。"
)

if run:
    df = scan_cash_or_margin_secured_puts(
        symbol=symbol,
        min_dte=min_dte,
        max_dte=max_dte,
        strike_min=(None if strike_min <= 0 else strike_min),
        strike_max=(None if strike_max <= 0 else strike_max),
        min_annualized=(None if min_ann <= 0 else min_ann),
    )

    if df.empty:
        st.warning("没有扫描到结果：可能是 DTE/strike 过滤太严格，或 Yahoo 数据暂时缺失。")
    else:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "下载 CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{symbol}_put_annualized_scan.csv",
            mime="text/csv",
        )

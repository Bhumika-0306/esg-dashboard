# ============================================================
# ESG INTELLIGENCE PLATFORM — app.py
# NEW: Beginner Mode toggle, plain-English tooltips,
#      Latest News + sentiment score in Tab 1,
#      FIXED: UI Alignment, Dark Mode text, and Tab State
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="ESG Intelligence Platform",
    page_icon="🌱",
    layout="wide"
)

# ── SESSION STATE: keeps beginner mode when switching tabs ───
if "beginner" not in st.session_state:
    st.session_state.beginner = False

# ════════════════════════════════════════════════════════════
# SIDEBAR — Beginner Mode + Glossary
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.session_state.beginner = st.toggle(
        "🎓 Beginner Mode",
        value=st.session_state.beginner,
        help="Replaces all financial jargon with plain English explanations"
    )
    if st.session_state.beginner:
        st.success("Beginner Mode ON — all jargon simplified!")
    else:
        st.info("Toggle Beginner Mode to simplify all labels")

    st.markdown("---")
    st.markdown("### 📖 Quick Glossary")

    with st.expander("📊 What is ESG Score?"):
        st.write(
            "ESG stands for **Environmental, Social, Governance**. "
            "It's a rating of how responsible and ethical a company is. "
            "Higher score = better company behaviour. Think of it as a "
            "'good citizen' score for businesses."
        )
    with st.expander("📈 What is Sharpe Ratio?"):
        st.write(
            "A measure of how much **return you get per unit of risk**. "
            "Above 1.0 = good. Above 2.0 = excellent. "
            "Below 0 = taking risk but losing money. "
            "Named after Nobel laureate William Sharpe."
        )
    with st.expander("📉 What is VaR / CVaR?"):
        st.write(
            "**Value at Risk (VaR):** The maximum you could lose in a month "
            "with 95% confidence. E.g., VaR = -5% means there's a 5% chance "
            "you lose MORE than 5%.\n\n"
            "**CVaR:** The average loss in those worst 5% scenarios. "
            "Always worse than VaR — it's the 'how bad could it get' number."
        )
    with st.expander("♻️ What is Greenwashing?"):
        st.write(
            "When a company **claims** to be environmentally responsible "
            "but their actual behaviour doesn't match. "
            "Our detector flags companies with high official ESG scores "
            "but negative market sentiment — a credibility gap."
        )
    with st.expander("🚨 What is an Anomaly?"):
        st.write(
            "A company-month where the financial data looks **very unusual** "
            "compared to historical patterns. Could signal earnings shocks, "
            "regulatory action, or hidden risk. Detected using IsolationForest ML."
        )
    with st.expander("🔄 What is Backtesting?"):
        st.write(
            "Testing an investment strategy on **historical data** to see "
            "how it would have performed in the past. Walk-forward validation "
            "means we train on old data and test on newer unseen data — "
            "preventing fake results."
        )
    with st.expander("📐 What is Walk-Forward Validation?"):
        st.write(
            "A rigorous testing method where the model is trained only on "
            "past data and tested on future data it has never seen. "
            "This prevents 'data snooping' — the model can't cheat by "
            "looking at future information."
        )
    with st.expander("🏦 What is the Efficient Market Hypothesis?"):
        st.write(
            "The theory that stock prices already reflect all available "
            "information — making it very hard to consistently beat the market. "
            "This is why our return prediction R² is close to zero, "
            "which is actually the expected result!"
        )

# ── LABEL HELPER ─────────────────────────────────────────────
beginner = st.session_state.beginner

def lbl(expert, simple):
    return simple if beginner else expert

# ── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_data():
    panel       = pd.read_csv("data/dynamic_panel.csv")
    ticker_sum  = pd.read_csv("data/ticker_summary.csv")
    gw_scores   = pd.read_csv("data/greenwashing_scores.csv")
    backtest    = pd.read_csv("data/backtest_monthly_returns.csv")
    bt_metrics  = pd.read_csv("data/backtest_metrics.csv")
    anomalies   = pd.read_csv("data/detected_anomalies.csv")
    ci_preds    = pd.read_csv("data/esg_predictions_with_ci.csv")
    var_summary = pd.read_csv("data/var_cvar_summary.csv")
    var_sims    = pd.read_csv("data/var_simulated_returns.csv")
    return (panel, ticker_sum, gw_scores, backtest, bt_metrics,
            anomalies, ci_preds, var_summary, var_sims)

@st.cache_resource
def load_models():
    feat_cols = joblib.load("models/feature_cols_m1.pkl")
    return {
        "esg":       joblib.load("models/model_fin.pkl"),
        "return":    joblib.load("models/model_return_predictor.pkl"),
        "risk_clf":  joblib.load("models/model_risk_classifier.pkl"),
        "anomaly":   joblib.load("models/model_anomaly_detector.pkl"),
        "feat_cols": feat_cols,
    }

(panel, ticker_sum, gw_scores, backtest, bt_metrics,
 anomalies, ci_preds, var_summary, var_sims) = load_data()
models       = load_models()
FEATURE_COLS = models["feat_cols"]
RISK_LABELS  = {0: "🟢 Low Risk",      1: "🟡 Medium Risk",    2: "🔴 High Risk"}
RISK_SIMPLE  = {0: "🟢 Safe Company",  1: "🟡 Average Risk",   2: "🔴 Risky Company"}

# ── HEADER ───────────────────────────────────────────────────
st.title("🌱 ESG Intelligence Platform")
if beginner:
    st.markdown(
        "A tool that analyses **how responsible and risky** companies are, "
        "using AI and financial data from 477 companies over 7 years (2018–2026)."
    )
else:
    st.markdown("Real-time ESG + financial risk analysis across 477 companies · 2018–2026")

# ── NAVIGATION (State-preserving radio buttons instead of tabs) ──
tab_names = [
    "🔍 " + lbl("Live Lookup",    "Look Up a Company"),
    "📊 " + lbl("Portfolio",      "Build Investment Mix"),
    "📈 " + lbl("Backtesting",    "Test Strategies"),
    "⚠️ " + lbl("Greenwashing",   "Spot Fake Claims"),
    "🚨 " + lbl("Anomalies",      "Unusual Alerts"),
    "💀 " + lbl("Risk Engine",    "Max Loss Calculator")
]

st.markdown("<br>", unsafe_allow_html=True)
active_tab = st.radio("Navigation", tab_names, horizontal=True, label_visibility="collapsed")
st.markdown("---")

# ════════════════════════════════════════════════════════════
# TAB 1: LIVE COMPANY LOOKUP
# ════════════════════════════════════════════════════════════
if active_tab == tab_names[0]:
    st.header(lbl("Live Company Analysis", "Look Up Any Company"))
    if beginner:
        st.markdown(
            "Type a stock ticker (e.g. **AAPL** for Apple, **TSLA** for Tesla) "
            "to see its ESG score, risk level, predicted return, and latest news."
        )

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        ticker_input = st.text_input(
            lbl("Stock Ticker", "Company Code (e.g. AAPL = Apple)"),
            value="AAPL"
        )
    with col_btn:
        # Pushes button down to align with the input box
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        run_lookup = st.button("Analyse", type="primary", use_container_width=True)

    if run_lookup and ticker_input:
        ticker_input = ticker_input.upper().strip()
        in_universe  = ticker_input in panel["Ticker"].unique()

        if not in_universe:
            st.warning(
                f"⚠️ {ticker_input} is not in the 477-ticker training universe. "
                "Predictions use live features only — accuracy may be lower."
            )

        with st.spinner(f"Fetching live data for {ticker_input}..."):
            try:
                # ── Download price data ───────────────────────
                raw = yf.download(
                    ticker_input, period="2y", interval="1d",
                    progress=False, auto_adjust=True
                )
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                if raw.empty:
                    st.error("No data found for this ticker.")
                    st.stop()

                raw = raw.reset_index()

                # ── Feature engineering ───────────────────────
                raw["DailyReturn"]     = raw["Close"].pct_change()
                raw["Return_7d"]       = raw["Close"].pct_change(7)
                raw["Return_14d"]      = raw["Close"].pct_change(14)
                raw["Return_30d"]      = raw["Close"].pct_change(30)
                raw["Volatility_14d"]  = raw["DailyReturn"].rolling(14).std()
                raw["Volatility_30d"]  = raw["DailyReturn"].rolling(30).std()
                raw["DownsideVol_30d"] = (
                    raw["DailyReturn"].where(raw["DailyReturn"] < 0, 0).rolling(30).std()
                )
                raw["Sharpe_30d"]  = (
                    raw["DailyReturn"].rolling(30).mean() /
                    (raw["Volatility_30d"] + 1e-9)
                )
                raw["MA_10"]       = raw["Close"].rolling(10).mean()
                raw["MA_30"]       = raw["Close"].rolling(30).mean()
                raw["MA_Spread"]   = (raw["MA_10"] - raw["MA_30"]) / (raw["MA_30"] + 1e-9)
                delta              = raw["Close"].diff()
                gain               = delta.where(delta > 0, 0).rolling(14).mean()
                loss               = (-delta.where(delta < 0, 0)).rolling(14).mean()
                raw["RSI_14"]      = 100 - (100 / (1 + gain / (loss + 1e-9)))
                log_vol            = np.log1p(raw["Volume"])
                raw["Volume_norm"] = (log_vol - log_vol.rolling(30).mean()) / (log_vol.rolling(30).mean() + 1e-9)
                bb_ma              = raw["Close"].rolling(20).mean()
                bb_std             = raw["Close"].rolling(20).std()
                raw["BB_Position"] = (raw["Close"] - (bb_ma - 2*bb_std)) / (4*bb_std + 1e-9)
                ema12              = raw["Close"].ewm(span=12).mean()
                ema26              = raw["Close"].ewm(span=26).mean()
                macd               = ema12 - ema26
                raw["MACD_Signal"]     = macd - macd.ewm(span=9).mean()
                raw["Volume_Surge"]    = raw["Volume"] / (raw["Volume"].rolling(20).mean() + 1e-9)
                raw["Momentum_Accel"]  = raw["Return_7d"] - raw["Return_30d"]

                if "High" in raw.columns and "Low" in raw.columns:
                    tr = pd.concat([
                        raw["High"] - raw["Low"],
                        (raw["High"] - raw["Close"].shift(1)).abs(),
                        (raw["Low"]  - raw["Close"].shift(1)).abs()
                    ], axis=1).max(axis=1)
                    raw["ATR_norm"] = tr.rolling(14).mean() / (raw["Close"] + 1e-9)
                else:
                    raw["ATR_norm"] = raw["Volatility_14d"]

                raw["Price_52w_High"]   = raw["Close"].rolling(252).max()
                raw["Price_52w_Low"]    = raw["Close"].rolling(252).min()
                raw["Position_52w"]     = (
                    (raw["Close"] - raw["Price_52w_Low"]) /
                    (raw["Price_52w_High"] - raw["Price_52w_Low"] + 1e-9)
                )
                raw["Return_12m_skip1"] = raw["Close"].shift(21).pct_change(231)
                vol6m                   = raw["Volatility_30d"].rolling(126).mean()
                raw["Vol_Regime"]       = raw["Volatility_30d"] / (vol6m + 1e-9)
                ma60                    = raw["Close"].rolling(60).mean()
                std60                   = raw["Close"].rolling(60).std()
                raw["MeanRev_Z"]        = (raw["Close"] - ma60) / (std60 + 1e-9)
                raw = raw.dropna()

                if len(raw) < 60:
                    st.error("Not enough data (need at least 60 trading days).")
                    st.stop()

                latest = raw.iloc[-1]
                X_list = []
                for f in FEATURE_COLS:
                    val = float(latest[f]) if f in latest.index else 0.0
                    if np.isnan(val) or np.isinf(val):
                        val = 0.0
                    X_list.append(val)
                X = np.array(X_list, dtype=np.float32).reshape(1, -1)

                esg_val   = (panel[panel["Ticker"]==ticker_input]["ESG_Score"].mean()
                             if in_universe else float(models["esg"].predict(X)[0]))
                ret_pred  = float(models["return"].predict(X)[0])
                risk_pred = int(models["risk_clf"].predict(X)[0])

                st.success(f"Analysis complete for **{ticker_input}**")

                # ── Core metrics ──────────────────────────────
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    lbl("ESG Score", "Responsibility Score"),
                    f"{esg_val:.0f}",
                    help=lbl(
                        "Average ESG score from training panel (higher = better)",
                        "How responsible this company is — higher is better (max ~1400)"
                    )
                )
                c2.metric(
                    lbl("ESG Risk Category", "Risk Level"),
                    RISK_SIMPLE[risk_pred] if beginner else RISK_LABELS[risk_pred],
                    help=lbl(
                        "3-class XGBoost classifier output",
                        "Based on how the company compares to all others in our dataset"
                    )
                )
                c3.metric(
                    lbl("Predicted 30d Return", "Expected Gain/Loss Next Month"),
                    f"{ret_pred*100:+.2f}%",
                    help=lbl(
                        "XGBoost return predictor output (low R² expected — EMH)",
                        "Our AI's best guess at next month's price change. Stock prediction is inherently uncertain!"
                    )
                )
                c4.metric(
                    lbl("RSI (14-day)", "Momentum Signal"),
                    f"{float(latest['RSI_14']):.0f}",
                    delta=(
                        lbl("Overbought", "Trending up too fast") if float(latest["RSI_14"]) > 70 else
                        lbl("Oversold",   "Possible buy signal")  if float(latest["RSI_14"]) < 30 else
                        "Neutral"
                    ),
                    help=lbl(
                        "RSI > 70 = overbought, < 30 = oversold",
                        "Above 70 = stock may be overpriced. Below 30 = may be underpriced."
                    )
                )

                # ── Market signals ────────────────────────────
                st.subheader(lbl("Market Signals", "More Indicators"))
                s1, s2, s3, s4 = st.columns(4)

                pos_52w   = float(latest.get("Position_52w",   np.nan))
                vol_reg   = float(latest.get("Vol_Regime",     np.nan))
                mean_rev  = float(latest.get("MeanRev_Z",      np.nan))
                mom_accel = float(latest.get("Momentum_Accel", np.nan))

                if not np.isnan(pos_52w):
                    s1.metric(
                        lbl("52-Week Position", "Where in Year Range?"),
                        f"{pos_52w*100:.0f}%",
                        delta="Near yearly high" if pos_52w > 0.8 else
                              "Near yearly low"  if pos_52w < 0.2 else "Mid-range",
                        help=lbl(
                            "% between 52-week low and high",
                            "100% = at its highest price of the year. 0% = at its lowest."
                        )
                    )
                if not np.isnan(vol_reg):
                    s2.metric(
                        lbl("Vol Regime", "Volatility Level"),
                        f"{vol_reg:.2f}x",
                        delta=lbl("Elevated risk", "More jumpy than usual") if vol_reg > 1.2 else "Normal",
                        delta_color="inverse" if vol_reg > 1.2 else "normal",
                        help=lbl(
                            "Current vol / 6-month avg vol. >1 = elevated",
                            "How much the stock is moving vs its usual behaviour."
                        )
                    )
                if not np.isnan(mean_rev):
                    s3.metric(
                        lbl("Mean Reversion Z", "Over/Under-Priced Signal"),
                        f"{mean_rev:.2f}",
                        delta=(
                            lbl("Oversold",   "May be underpriced") if mean_rev < -1.5 else
                            lbl("Overbought", "May be overpriced")  if mean_rev >  1.5 else
                            "Neutral"
                        ),
                        help=lbl(
                            "Z-score vs 60-day moving average. |Z| > 1.5 = notable deviation",
                            "How far price is from its recent average."
                        )
                    )
                if not np.isnan(mom_accel):
                    s4.metric(
                        lbl("Momentum Accel", "Speed of Price Change"),
                        f"{mom_accel*100:+.2f}%",
                        delta=(
                            lbl("Accelerating", "Gaining momentum") if mom_accel >  0.01 else
                            lbl("Decelerating", "Losing momentum")  if mom_accel < -0.01 else
                            "Flat"
                        ),
                        help=lbl("7d return minus 30d return",
                                 "Is the stock speeding up or slowing down?")
                    )

                # ── Price chart ───────────────────────────────
                st.subheader(f"{ticker_input} — " + lbl("Price (Last 12 Months)", "Share Price History"))
                chart_raw = raw.tail(252)
                fig, ax   = plt.subplots(figsize=(11, 3))
                ax.plot(chart_raw["Date"], chart_raw["Close"], color="#378ADD", linewidth=1.5)
                ax.fill_between(chart_raw["Date"], chart_raw["Close"],
                                chart_raw["Close"].min(), alpha=0.08, color="#378ADD")
                ax.set_ylabel("Price (USD)")
                ax.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # ── Latest News + Sentiment ───────────────────
                st.subheader(
                    lbl(f"Latest News — {ticker_input}",
                        f"What's in the News for {ticker_input}?")
                )
                if beginner:
                    st.caption(
                        "We score each headline from **-1 (very negative)** to "
                        "**+1 (very positive)** based on the financial language used."
                    )

                try:
                    ticker_obj = yf.Ticker(ticker_input)
                    news_data  = ticker_obj.news

                    if not news_data:
                        st.info("No recent news found for this ticker.")
                    else:
                        pos_words = ["gain","rise","surge","beat","strong","growth",
                                     "profit","up","high","record","buy","upgrade",
                                     "positive","bullish","exceed","outperform","rally","soar"]
                        neg_words = ["fall","drop","miss","weak","loss","down","low",
                                     "cut","downgrade","risk","concern","warn","negative",
                                     "bearish","decline","crash","fear","sell","layoff"]

                        news_rows = []
                        for item in news_data[:8]:
                            content = item.get("content", item)

                            title = (
                                content.get("title") or
                                item.get("title") or
                                content.get("headline") or
                                ""
                            )
                            publisher = (
                                content.get("provider", {}).get("displayName") or
                                content.get("source") or
                                item.get("publisher") or
                                "Unknown"
                            )
                            link = (
                                content.get("canonicalUrl", {}).get("url") or
                                content.get("url") or
                                item.get("link") or
                                "#"
                            )

                            if not title:
                                continue

                            title_lower = title.lower()
                            pos_count = sum(1 for w in pos_words if w in title_lower)
                            neg_count = sum(1 for w in neg_words if w in title_lower)
                            total_w   = pos_count + neg_count

                            if total_w == 0:
                                score      = 0.0
                                label_sent = "⚪ Neutral"
                            elif pos_count > neg_count:
                                score      = round(pos_count / total_w, 2)
                                label_sent = "🟢 Positive"
                            else:
                                score      = round(-neg_count / total_w, 2)
                                label_sent = "🔴 Negative"

                            news_rows.append({
                                "Headline":  title,
                                "Source":    publisher,
                                "Sentiment": label_sent,
                                "Score":     score,
                                "Link":      link
                            })

                        if news_rows:
                            news_df = pd.DataFrame(news_rows)
                            for _, nr in news_df.iterrows():
                                col_news, col_score = st.columns([5, 1])
                                with col_news:
                                    st.markdown(
                                        f"**[{nr['Headline']}]({nr['Link']})** "
                                        f"<span style='color:gray;font-size:0.8rem'>"
                                        f"{nr['Source']}</span>",
                                        unsafe_allow_html=True
                                    )
                                with col_score:
                                    score_color = (
                                        "#1D9E75" if nr["Score"] > 0 else
                                        "#E24B4A" if nr["Score"] < 0 else
                                        "#888780"
                                    )
                                    st.markdown(
                                        f"<div style='text-align:center;"
                                        f"background:{score_color}20;"
                                        f"border-left:3px solid {score_color};"
                                        f"border-radius:6px;padding:0.4rem;"
                                        f"font-weight:600;color:{score_color}'>"
                                        f"{nr['Sentiment']}<br>"
                                        f"<span style='font-size:1.1rem'>"
                                        f"{nr['Score']:+.2f}</span></div>",
                                        unsafe_allow_html=True
                                    )
                                st.markdown("---")

                            avg_score = news_df["Score"].mean()
                            sc1, sc2, sc3 = st.columns(3)
                            sc1.metric(
                                lbl("Avg Sentiment Score", "Average News Mood"),
                                f"{avg_score:+.2f}"
                            )
                            sc2.metric(
                                lbl("Positive Headlines", "Good News"),
                                f"{(news_df['Score']>0).sum()} / {len(news_df)}"
                            )
                            sc3.metric(
                                lbl("Negative Headlines", "Bad News"),
                                f"{(news_df['Score']<0).sum()} / {len(news_df)}"
                            )
                        else:
                            st.info("Could not parse news headlines for this ticker.")

                except Exception as news_err:
                    st.info(f"News temporarily unavailable: {str(news_err)}")

                # ── Panel history ─────────────────────────────
                if in_universe:
                    with st.expander(
                        lbl(
                            f"Training panel history ({panel[panel['Ticker']==ticker_input].shape[0]} months)",
                            f"Historical data used to train our AI ({panel[panel['Ticker']==ticker_input].shape[0]} months)"
                        )
                    ):
                        hist = (panel[panel["Ticker"]==ticker_input]
                                .sort_values("YearMonth").tail(12)
                                .reset_index(drop=True))
                        show_cols = (["YearMonth","ESG_Score"] +
                                     [c for c in FEATURE_COLS if c in hist.columns][:8])
                        st.dataframe(
                            hist[[c for c in show_cols if c in hist.columns]],
                            use_container_width=True
                        )

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ════════════════════════════════════════════════════════════
# TAB 2: PORTFOLIO BUILDER
# ════════════════════════════════════════════════════════════
elif active_tab == tab_names[1]:
    st.header(lbl("ESG-Weighted Portfolio Builder", "Build Your Investment Mix"))
    if beginner:
        st.markdown(
            "Choose companies and decide how much you care about ESG vs returns. "
            "We'll calculate the ideal mix for you."
        )

    all_tickers = sorted(ticker_sum["Ticker"].unique().tolist())

    if "Regime" in panel.columns:
        regimes    = ["All"] + sorted(panel["Regime"].dropna().unique().tolist())
        sel_regime = st.selectbox(
            lbl("Filter by market regime", "Show companies from a specific market period"),
            regimes,
            help=lbl(
                "Only show tickers that appeared in this regime",
                "Filter to see companies from specific market conditions like COVID or the AI boom"
            )
        )
        if sel_regime != "All":
            regime_tickers = panel[panel["Regime"]==sel_regime]["Ticker"].unique().tolist()
            all_tickers    = [t for t in all_tickers if t in regime_tickers]

    col_sel, col_alpha = st.columns([3, 1])
    with col_sel:
        selected = st.multiselect(
            lbl("Select tickers", "Choose companies"),
            all_tickers,
            default=all_tickers[:8] if len(all_tickers) >= 8 else all_tickers
        )
    with col_alpha:
        alpha = st.slider(
            lbl("ESG weight (α)", "How much to prioritise ethics vs profit"),
            0.0, 1.0, 0.6, 0.1,
            help=lbl(
                "1.0 = pure ESG weighting. 0.0 = pure return weighting",
                "Slide right to favour ethical companies. Slide left to favour profitable ones."
            )
        )

    if selected:
        port_df = ticker_sum[ticker_sum["Ticker"].isin(selected)].copy()
        for col in ["ESG_Score","Predicted_Return"]:
            mn, mx = port_df[col].min(), port_df[col].max()
            port_df[f"{col}_norm"] = (port_df[col] - mn) / (mx - mn + 1e-9)
        port_df["Combined_Score"] = (
            alpha * port_df["ESG_Score_norm"] +
            (1 - alpha) * port_df["Predicted_Return_norm"]
        )
        port_df["Weight"] = port_df["Combined_Score"] / port_df["Combined_Score"].sum()
        port_df = port_df.sort_values("Weight", ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(lbl("**Portfolio Allocation**", "**How money is split**"))
            fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
            colors_pie = plt.cm.Set3(np.linspace(0, 1, len(port_df)))
            ax_pie.pie(
                port_df["Weight"], labels=port_df["Ticker"],
                autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
                colors=colors_pie, startangle=140, pctdistance=0.8
            )
            ax_pie.set_title(
                lbl(f"Portfolio Weights (α={alpha})",
                    f"Your Investment Split (Ethics weight={int(alpha*100)}%)")
            )
            st.pyplot(fig_pie, use_container_width=True)
            plt.close()

        with c2:
            st.markdown(lbl("**Company Scorecard**", "**Company Report Card**"))
            disp_cols = [c for c in
                         ["Ticker","ESG_Score","Predicted_Return","Sharpe_30d","Weight"]
                         if c in port_df.columns]
            port_display = port_df[disp_cols].copy()
            if beginner:
                port_display = port_display.rename(columns={
                    "Ticker":           "Company",
                    "ESG_Score":        "Ethics Score",
                    "Predicted_Return": "Expected Return",
                    "Sharpe_30d":       "Risk-Adjusted Score",
                    "Weight":           "Your Allocation"
                })
            port_display.iloc[:,1] = port_display.iloc[:,1].apply(lambda x: f"{float(x):.0f}")
            port_display.iloc[:,2] = port_display.iloc[:,2].apply(lambda x: f"{float(x)*100:+.2f}%")
            port_display.iloc[:,3] = port_display.iloc[:,3].apply(lambda x: f"{float(x):.3f}")
            port_display.iloc[:,4] = port_display.iloc[:,4].apply(lambda x: f"{float(x)*100:.1f}%")
            st.dataframe(port_display.reset_index(drop=True), use_container_width=True)

            st.markdown(lbl("**Portfolio Metrics**", "**Overall Summary**"))
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(lbl("Weighted ESG Score",   "Avg Ethics Score"),
                       f"{(port_df['ESG_Score']*port_df['Weight']).sum():.0f}")
            mc2.metric(lbl("Avg Predicted Return", "Expected Monthly Gain"),
                       f"{(port_df['Predicted_Return']*port_df['Weight']).sum()*100:+.2f}%")
            mc3.metric(lbl("Weighted Sharpe",      "Risk-Adjusted Quality"),
                       f"{(port_df['Sharpe_30d']*port_df['Weight']).sum():.3f}",
                       help=lbl("Weighted avg Sharpe",
                                "Higher = better returns for the risk taken"))

# ════════════════════════════════════════════════════════════
# TAB 3: BACKTESTING ENGINE
# ════════════════════════════════════════════════════════════
elif active_tab == tab_names[2]:
    st.header(lbl("Strategy Backtesting Engine", "How Would These Strategies Have Done?"))
    if beginner:
        st.markdown(
            "We test 4 different investment strategies on **real historical data** "
            "to see which one performs best. The model was trained on 2018–2023 data "
            "and tested on 2024–2026 data it had never seen before."
        )

    strat_map = {
        "EqualWeight": (lbl("Equal Weight",  "Spread Equally"), "#888780"),
        "ESGWeighted": (lbl("ESG-Weighted",  "Ethical Focus"),  "#1D9E75"),
        "Momentum":    (lbl("Momentum",      "Follow Winners"), "#378ADD"),
        "Blend":       (lbl("ESG-Mom Blend", "Best of Both"),   "#c0621a"),
    }

    bq1, bq2, bq3 = st.columns(3)
    bq1.metric(lbl("Walk-forward split", "How data was split"), "70% train / 30% test")
    bq2.metric(lbl("Train period",       "Learning period"),    "2018–2023")
    bq3.metric(lbl("Test period",        "Testing period"),     "2024–2026")

    st.markdown("### " + lbl("Performance Summary", "Strategy Results"))
    cols_bt = st.columns(len(bt_metrics))
    for i, (_, row) in enumerate(bt_metrics.iterrows()):
        total   = row.get("Total_%",      row.get("Total_Return",  0))
        ann_ret = row.get("Ann_Return_%", total)
        sharpe  = row.get("Sharpe",       0)
        max_dd  = row.get("MaxDD_%",      row.get("MaxDrawdown",   0))
        win_r   = row.get("WinRate_%",    row.get("WinRate",       0))
        ann_vol = row.get("Ann_Vol_%",    0)
        color   = "#888780"
        for key, c in [("ESG","#1D9E75"),("Momentum","#378ADD"),("Blend","#c0621a")]:
            if key.lower() in str(row["Strategy"]).lower():
                color = c
                break
        with cols_bt[i]:
            if beginner:
                st.markdown(f"""
                <div style="background:#f8f9fa; color:#1e1e1e; border-radius:10px;padding:1rem;
                            border-left:4px solid {color};margin-bottom:0.5rem;">
                    <strong>{row['Strategy']}</strong><br>
                    Total Profit/Loss: <b>{total:+.1f}%</b><br>
                    Yearly Return: <b>{ann_ret:+.1f}%</b><br>
                    Risk Quality: <b>{sharpe:.2f}</b>
                    <span style="color:gray;font-size:0.8rem">(higher=better)</span><br>
                    Worst Month Loss: <b>{max_dd:.1f}%</b><br>
                    Winning Months: <b>{win_r:.0f}%</b>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#f8f9fa; color:#1e1e1e; border-radius:10px;padding:1rem;
                            border-left:4px solid {color};margin-bottom:0.5rem;">
                    <strong>{row['Strategy']}</strong><br>
                    Total Return: <b>{total:+.1f}%</b><br>
                    Ann. Return: <b>{ann_ret:+.1f}%</b><br>
                    Ann. Vol: <b>{ann_vol:.1f}%</b><br>
                    Sharpe: <b>{sharpe:.3f}</b><br>
                    Max Drawdown: <b>{max_dd:.1f}%</b><br>
                    Win Rate: <b>{win_r:.1f}%</b>
                </div>""", unsafe_allow_html=True)

    st.markdown("### " + lbl("Cumulative Returns", "Portfolio Growth Over Time"))
    fig_bt, ax_bt = plt.subplots(figsize=(11, 4))
    for key, (label, color) in strat_map.items():
        col = f"{key}_Cumulative"
        if col in backtest.columns:
            ax_bt.plot(range(len(backtest)), backtest[col],
                       label=label, color=color, linewidth=2.2)
    ax_bt.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4, label="Breakeven")
    step = max(1, len(backtest) // 8)
    ax_bt.set_xticks(range(0, len(backtest), step))
    ax_bt.set_xticklabels(backtest["YearMonth"].iloc[::step].tolist(), rotation=30, fontsize=8)
    ax_bt.set_ylabel(lbl("Portfolio Value (₹1 invested)", "Value of ₹1 invested"))
    ax_bt.legend(fontsize=9)
    ax_bt.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_bt, use_container_width=True)
    plt.close()

    st.markdown("### " + lbl("Monthly Returns by Strategy", "Month-by-Month Results"))
    fig_mb, ax_mb = plt.subplots(figsize=(11, 4))
    x = np.arange(len(backtest))
    w = 0.2
    for i, (key, (label, color)) in enumerate(strat_map.items()):
        col = f"{key}_Return"
        if col in backtest.columns:
            ax_mb.bar(x + i*w, backtest[col]*100, width=w,
                      label=label, color=color, alpha=0.8)
    ax_mb.axhline(0, color="black", linewidth=0.7)
    ax_mb.set_ylabel(lbl("Return (%)", "Gain or Loss (%)"))
    ax_mb.set_xticks(x + w*1.5)
    ax_mb.set_xticklabels(backtest["YearMonth"].tolist(), rotation=45, ha="right", fontsize=7)
    ax_mb.legend(fontsize=9)
    ax_mb.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    st.pyplot(fig_mb, use_container_width=True)
    plt.close()

    st.markdown("### " + lbl("Sharpe Ratio Comparison", "Risk vs Reward Score"))
    fig_sh, ax_sh = plt.subplots(figsize=(7, 3))
    sharpes_sh = [r.get("Sharpe", 0) for _, r in bt_metrics.iterrows()]
    labels_sh  = [r["Strategy"] for _, r in bt_metrics.iterrows()]
    bc_sh = []
    for _, r in bt_metrics.iterrows():
        c = "#888780"
        for key, col in [("ESG","#1D9E75"),("Momentum","#378ADD"),("Blend","#c0621a")]:
            if key.lower() in str(r["Strategy"]).lower():
                c = col
                break
        bc_sh.append(c)
    bars_sh = ax_sh.bar(labels_sh, sharpes_sh, color=bc_sh, alpha=0.85)
    ax_sh.axhline(1.0, color="red", linewidth=1, linestyle="--", alpha=0.5,
                  label=lbl("Sharpe=1 threshold", "Good performance line"))
    ax_sh.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    for bar, v in zip(bars_sh, sharpes_sh):
        ax_sh.text(bar.get_x()+bar.get_width()/2,
                   bar.get_height()+0.01 if v >= 0 else bar.get_height()-0.06,
                   f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax_sh.set_ylabel(lbl("Sharpe Ratio", "Risk-Adjusted Score"))
    ax_sh.set_title(lbl("Risk-Adjusted Performance", "How good is each strategy per unit of risk?"))
    ax_sh.legend(fontsize=9)
    ax_sh.grid(alpha=0.3, axis="y")
    ax_sh.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    st.pyplot(fig_sh, use_container_width=True)
    plt.close()

    if beginner:
        st.info(
            "📌 **Plain English:** The Momentum strategy had the best risk-adjusted score. "
            "The ESG strategy underperformed slightly in this short window — that's normal. "
            "ESG benefits usually take years to show up, not months."
        )
    else:
        st.info(
            "📌 **Methodology:** XGBoost on 20 dynamic features. Forward returns as P&L. "
            "Top-10 stocks per strategy, 20% max position, 10bps transaction cost. "
            "Low R² is expected — consistent with EMH."
        )

# ════════════════════════════════════════════════════════════
# TAB 4: GREENWASHING RADAR
# ════════════════════════════════════════════════════════════
elif active_tab == tab_names[3]:
    st.header(lbl("Greenwashing Detection Radar", "Spot Fake Green Claims"))
    if beginner:
        st.markdown(
            "Some companies claim to be environmentally responsible but "
            "the market doesn't believe them. We flag companies where the "
            "**official ESG score** is much higher than what news sentiment suggests."
        )
    else:
        st.markdown(
            "Companies where the **official ESG score** significantly exceeds "
            "**market-derived news sentiment** — a potential ESG credibility gap signal."
        )

    gw_counts  = gw_scores["GW_Category"].value_counts()
    cat_colors = {"High Risk":"#E24B4A","Moderate Risk":"#EF9F27",
                  "Aligned":"#1D9E75","Outperformer":"#185FA5"}
    cat_simple = {
        "High Risk":     "⚠️ Possibly Misleading",
        "Moderate Risk": "🟡 Some Concern",
        "Aligned":       "✅ Claims Match Reality",
        "Outperformer":  "🌟 Better Than Claimed"
    }

    col_gw = st.columns(4)
    for i, (cat, color) in enumerate(cat_colors.items()):
        display_cat = cat_simple[cat] if beginner else cat
        col_gw[i].markdown(f"""
        <div style="background:#f8f9fa; color:#1e1e1e; border-radius:10px;padding:1rem;
                    border-left:4px solid {color};margin-bottom:0.5rem;">
            <strong>{display_cat}</strong><br>
            <span style="font-size:1.8rem;font-weight:600">{gw_counts.get(cat,0)}</span>
            <span style="color:gray"> companies</span>
        </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(lbl("**Top 10 Greenwashing-Risk Companies**",
                        "**Top 10 Companies With Suspicious ESG Claims**"))
        top_gw = gw_scores.nlargest(10, "Greenwashing_Index")[
            ["Ticker","ESG_Score","Avg_Sentiment","Greenwashing_Index","GW_Category"]
        ].reset_index(drop=True)
        top_gw["ESG_Score"]          = top_gw["ESG_Score"].apply(lambda x: f"{x:.0f}")
        top_gw["Avg_Sentiment"]      = top_gw["Avg_Sentiment"].apply(lambda x: f"{x:.4f}")
        top_gw["Greenwashing_Index"] = top_gw["Greenwashing_Index"].apply(lambda x: f"{x:.4f}")
        st.dataframe(top_gw, use_container_width=True)

    with c2:
        st.markdown(lbl("**ESG Score vs Sentiment Map**",
                        "**Official Score vs What Markets Think**"))
        fig_gw, ax_gw = plt.subplots(figsize=(6, 5))
        for category, group in gw_scores.groupby("GW_Category"):
            ax_gw.scatter(group["ESG_Score_norm"], group["Sentiment_norm"],
                          c=cat_colors.get(category,"#888780"), label=category,
                          alpha=0.7, s=50, edgecolors="white", linewidth=0.4)
        ax_gw.plot([0,1],[0,1],"k--",linewidth=1,alpha=0.3,label="Perfect alignment")
        ax_gw.set_xlabel(lbl("ESG Score (normalised)", "Official Ethics Score"))
        ax_gw.set_ylabel(lbl("News Sentiment (normalised)", "Market Opinion"))
        ax_gw.legend(fontsize=8)
        ax_gw.set_title(lbl("Greenwashing Risk Map",
                            "Above the line = market opinion lower than official score"))
        plt.tight_layout()
        st.pyplot(fig_gw, use_container_width=True)
        plt.close()

# ════════════════════════════════════════════════════════════
# TAB 5: ANOMALY MONITOR
# ════════════════════════════════════════════════════════════
elif active_tab == tab_names[4]:
    st.header(lbl("Anomaly Monitor", "Unusual Company Alerts"))
    if beginner:
        st.markdown(
            "Our AI flags company-months that look **very different** from normal. "
            "This could mean a big earnings miss, a scandal, or an unusual market event."
        )

    st.metric(
        lbl("Total Anomalies Detected", "Unusual Situations Found"),
        len(anomalies),
        help=lbl("~5% contamination rate — IsolationForest on 9 financial features",
                 "Out of all company-months, these looked unusually different")
    )

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown(lbl("**Most Anomalous Company-Months**", "**Most Unusual Situations**"))
        show_cols = [c for c in
                     ["Ticker","YearMonth","ESG_Score","DailyReturn","Volatility_30d","Sharpe_30d"]
                     if c in anomalies.columns]
        anom_display = anomalies.nsmallest(20,"Anomaly_Raw")[show_cols].reset_index(drop=True)
        if "DailyReturn"    in anom_display.columns:
            anom_display["DailyReturn"]    = anom_display["DailyReturn"].apply(lambda x: f"{x*100:+.2f}%")
        if "Volatility_30d" in anom_display.columns:
            anom_display["Volatility_30d"] = anom_display["Volatility_30d"].apply(lambda x: f"{x:.4f}")
        if "Sharpe_30d"     in anom_display.columns:
            anom_display["Sharpe_30d"]     = anom_display["Sharpe_30d"].apply(lambda x: f"{x:.3f}")
        if beginner:
            anom_display = anom_display.rename(columns={
                "Ticker":"Company","YearMonth":"Month","ESG_Score":"Ethics Score",
                "DailyReturn":"Daily Change","Volatility_30d":"Price Swings","Sharpe_30d":"Risk Quality"
            })
        st.dataframe(anom_display, use_container_width=True)

    with col_a2:
        st.markdown(lbl("**Most Frequently Anomalous Companies**",
                        "**Companies With Most Unusual Months**"))
        anom_freq = anomalies["Ticker"].value_counts().head(15).reset_index()
        anom_freq.columns = ["Ticker","Anomaly Count"]
        fig_af, ax_af = plt.subplots(figsize=(6, 5))
        ax_af.barh(anom_freq["Ticker"][::-1], anom_freq["Anomaly Count"][::-1],
                   color="#E24B4A", alpha=0.8)
        ax_af.set_xlabel(lbl("Number of Anomalous Months", "How many unusual months"))
        ax_af.set_title(lbl("Most Frequently Anomalous Companies",
                            "Companies That Triggered the Most Alerts"))
        plt.tight_layout()
        st.pyplot(fig_af, use_container_width=True)
        plt.close()

    panel_anom = panel.merge(
        anomalies[["Ticker","YearMonth"]].assign(is_anomaly=1),
        on=["Ticker","YearMonth"], how="left"
    )
    panel_anom["is_anomaly"] = panel_anom["is_anomaly"].fillna(0)
    st.markdown(lbl("**ESG Score Distribution: Normal vs Anomalous**",
                    "**Do Unusual Companies Have Different Ethics Scores?**"))
    fig_dist, ax_dist = plt.subplots(figsize=(10, 3))
    ax_dist.hist(panel_anom[panel_anom["is_anomaly"]==0]["ESG_Score"],
                 bins=40, density=True, alpha=0.6, color="#1D9E75",
                 label=lbl("Normal","Normal Companies"))
    ax_dist.hist(panel_anom[panel_anom["is_anomaly"]==1]["ESG_Score"],
                 bins=20, density=True, alpha=0.8, color="#E24B4A",
                 label=lbl("Anomalous","Unusual Companies"))
    ax_dist.set_xlabel(lbl("ESG Score","Ethics Score"))
    ax_dist.set_ylabel("Density")
    ax_dist.legend()
    ax_dist.set_title(lbl("ESG Score: Anomalous vs Normal Company-Months",
                          "Do unusual companies tend to have lower ethics scores?"))
    plt.tight_layout()
    st.pyplot(fig_dist, use_container_width=True)
    plt.close()

# ════════════════════════════════════════════════════════════
# TAB 6: VaR / CVaR RISK ENGINE
# ════════════════════════════════════════════════════════════
elif active_tab == tab_names[5]:
    st.header(lbl("💀 VaR / CVaR Risk Engine", "💀 Maximum Loss Calculator"))
    if beginner:
        st.markdown(
            "We run **10,000 random simulations** of what could happen to each "
            "portfolio over the next month. This tells us the worst-case loss "
            "you should be prepared for."
        )
    else:
        st.markdown(
            "Monte Carlo simulation (10,000 scenarios) estimates maximum expected "
            "loss for each portfolio strategy over a **1-month horizon** at "
            "95% and 99% confidence using Geometric Brownian Motion."
        )

    colors_var = {"Equal Weight":"#888780","ESG-Weighted":"#1D9E75",
                  "Momentum":"#378ADD","ESG-Mom Blend":"#c0621a"}

    st.markdown("### " + lbl("Risk Summary", "How Much Could You Lose?"))
    cols_v      = st.columns(len(var_summary))
    var_row_map = {row["Strategy"]: row for _, row in var_summary.iterrows()}

    for i, (_, row) in enumerate(var_summary.iterrows()):
        color = colors_var.get(row["Strategy"],"#888780")
        with cols_v[i]:
            if beginner:
                st.markdown(f"""
                <div style="background:#f8f9fa; color:#1e1e1e; border-radius:10px;padding:1rem;
                            border-left:4px solid {color};margin-bottom:0.5rem;">
                    <strong style="font-size:0.85rem">{row['Strategy']}</strong><br>
                    Expected gain: <b>{row['Expected_%']:+.2f}%</b><br>
                    Typical swing: <b>±{row['Std_%']:.1f}%</b><br>
                    Worst 5% loss: <b style="color:#E24B4A">{row['VaR_95_%']:.1f}%</b><br>
                    Avg in worst 5%: <b style="color:#c0351a">{row['CVaR_95_%']:.1f}%</b>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#f8f9fa; color:#1e1e1e; border-radius:10px;padding:1rem;
                            border-left:4px solid {color};margin-bottom:0.5rem;">
                    <strong style="font-size:0.85rem">{row['Strategy']}</strong><br>
                    Exp. Return: <b>{row['Expected_%']:+.2f}%</b><br>
                    Std Dev: <b>{row['Std_%']:.2f}%</b><br>
                    95% VaR: <b style="color:#E24B4A">{row['VaR_95_%']:.2f}%</b><br>
                    95% CVaR: <b style="color:#c0351a">{row['CVaR_95_%']:.2f}%</b><br>
                    99% VaR: <b style="color:#8B0000">{row['VaR_99_%']:.2f}%</b><br>
                    MC Sharpe: <b>{row['MC_Sharpe']:.3f}</b>
                </div>""", unsafe_allow_html=True)

    if beginner:
        st.markdown("""
        > 💡 **Example:** If 95% VaR = **-5%**, it means in 95 out of 100 months
        > you won't lose more than 5%. But in the worst 5 months, you might lose more.
        > CVaR tells you how much more — it's the average of those worst cases.
        """)
    else:
        st.markdown("""
        > 📌 **How to read:** A 95% VaR of **-5%** means there is a **5% chance**
        > the portfolio loses more than 5% in the next month.
        > CVaR is the *average loss* in that worst 5% of scenarios.
        """)

    st.markdown("### " + lbl("Simulated Return Distributions", "Spread of Possible Outcomes"))
    fig_d, ax_d = plt.subplots(figsize=(11, 4))
    for strat, color in colors_var.items():
        sims = var_sims[var_sims["Strategy"]==strat]["Simulated_Return"]*100
        if len(sims) == 0:
            continue
        ax_d.hist(sims, bins=100, alpha=0.35, color=color, density=True, label=strat)
        if strat in var_row_map:
            var95 = var_row_map[strat]["VaR_95_%"]
            ax_d.axvline(var95, color=color, linewidth=2, linestyle="--")
    ax_d.axvline(0, color="black", linewidth=1, alpha=0.6)
    ax_d.set_xlabel(lbl("1-Month Portfolio Return (%)", "Possible monthly gain or loss (%)"))
    ax_d.set_ylabel("Density")
    ax_d.set_title(lbl("Monte Carlo Return Distributions (dashed = 95% VaR)",
                       "10,000 simulated outcomes — dashed line = worst 5% boundary"))
    ax_d.legend(fontsize=8, ncol=2)
    ax_d.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_d, use_container_width=True)
    plt.close()

    c1, c2 = st.columns(2)
    with c1:
        fig_v, ax_v = plt.subplots(figsize=(6, 4))
        strats_v = var_summary["Strategy"].tolist()
        x_v      = np.arange(len(strats_v))
        w_v      = 0.35
        bc_v     = [colors_var.get(s,"#888780") for s in strats_v]
        b1 = ax_v.bar(x_v-w_v/2, var_summary["VaR_95_%"],  w_v, color=bc_v, alpha=0.9, label="95% VaR")
        b2 = ax_v.bar(x_v+w_v/2, var_summary["CVaR_95_%"], w_v, color=bc_v, alpha=0.5,
                      edgecolor=bc_v, linewidth=1.5, label="95% CVaR")
        for bar, v in zip(b1, var_summary["VaR_95_%"]):
            ax_v.text(bar.get_x()+bar.get_width()/2,
                      bar.get_height()+(0.05 if v>=0 else -0.15),
                      f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax_v.set_xticks(x_v)
        ax_v.set_xticklabels([s.replace(" ","\n") for s in strats_v], fontsize=8)
        ax_v.set_title(lbl("95% VaR vs CVaR by Strategy", "Worst-Case Loss Comparison"))
        ax_v.legend(fontsize=9)
        ax_v.axhline(0, color="black", linewidth=1)
        ax_v.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        st.pyplot(fig_v, use_container_width=True)
        plt.close()

    with c2:
        fig_rs, ax_rs = plt.subplots(figsize=(6, 4))
        for _, row in var_summary.iterrows():
            color = colors_var.get(row["Strategy"],"#888780")
            ax_rs.scatter(row["Std_%"], row["Expected_%"], s=150,
                          color=color, zorder=5, edgecolors="white", linewidth=1)
            ax_rs.annotate(row["Strategy"], (row["Std_%"], row["Expected_%"]),
                           textcoords="offset points", xytext=(6,4), fontsize=8)
        ax_rs.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_rs.set_xlabel(lbl("Portfolio Std Dev (%)", "How much it swings (%)"))
        ax_rs.set_ylabel(lbl("Expected Return (%)", "Expected gain (%)"))
        ax_rs.set_title(lbl("Monte Carlo Risk–Return Scatter",
                            "More to the right = riskier. Higher up = better return."))
        ax_rs.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_rs, use_container_width=True)
        plt.close()

    st.markdown("### " + lbl("Full Metrics Table", "All Numbers"))
    st.dataframe(var_summary, use_container_width=True)

    if beginner:
        st.info(
            "💡 **How this works:** We simulate 10,000 possible futures using a "
            "mathematical model that captures how volatile each stock has been "
            "historically, then look at the worst outcomes to estimate max losses."
        )
    else:
        st.info(
            "📌 **Methodology:** GBM with per-ticker σ = Volatility_30d × √21. "
            "Equal-weight top-10 portfolios. 10,000 simulations, 1-month horizon."
        )

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>ESG Intelligence Platform · Capstone Project · "
    "Built with Streamlit, XGBoost, IsolationForest, Monte Carlo VaR · "
    "477 companies · 2018–2026</small></center>",
    unsafe_allow_html=True
)
# 🎯 AI Vietlott 5/35 – v2_auto (Dual-Mode + Auto-Update Daily + Vietlott UI)
# Run: streamlit run app.py

import streamlit as st
import requests, re, time, os, json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup

VN_TZ = timezone(timedelta(hours=7))
DATA_CSV = "lotto_5_35_history.csv"
META_JSON = "app_meta.json"

st.set_page_config(
    page_title="AI Vietlott 5/35 – Dual-Mode (Auto)",
    page_icon="🎯",
    layout="wide"
)

# ---------------------- UI header ---------------------- #
VTL_RED = "#E10600"
VTL_YELLOW = "#F5C400"

st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;padding:14px 10px;
                border-radius:14px;background:linear-gradient(90deg,{VTL_RED}0D, #ffffff 40%);">
        <img src="https://raw.githubusercontent.com/chienbt/vietlott-ai/main/logo_vietlott.svg" alt="Vietlott" height="48" />
        <div>
            <div style="font-weight:800;font-size:28px; line-height:1; color:#111;">
                AI Vietlott 5/35 – Dual-Mode Predictor (Hot + Stable)
            </div>
            <div style="color:#444; font-size:14px; margin-top:4px">
                Phân tích song song cửa sổ <b>NGẮN</b> (bắt trend) + <b>DÀI</b> (xác suất bền). Tự động cập nhật theo ngày.
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------------- Data update ---------------------- #
def retry_get(url, params=None, max_tries=5, backoff=0.8, timeout=12):
    for i in range(max_tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(backoff*(i+1))
    raise RuntimeError("Network error fetching data.")

def fetch_lotto8(pages=20, sleep=0.12):
    base = "https://www.lotto-8.com/Vietnam/listltoVM35.asp"
    rows=[]
    for p in range(1, pages+1):
        r = retry_get(base, params={"page":p})
        soup = BeautifulSoup(r.text, "html.parser")
        for tr in soup.select("table tr"):
            txt = tr.get_text(" ", strip=True)
            nums = [int(x) for x in re.findall(r'\b([0-2]?\d|3[0-5])\b', txt)]
            if len(nums) < 6: continue
            mains = [n for n in nums if 1 <= n <= 35]
            sps   = [n for n in nums if 1 <= n <= 12]
            mains = sorted(list(dict.fromkeys(mains)))[:5]
            if len(mains) != 5 or not sps: continue
            sp = sps[-1]
            m_date = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', txt)
            date = ""
            if m_date:
                y, mo, d = map(int, m_date.groups())
                date = f"{y:04d}-{mo:02d}-{d:02d}"
            rows.append({"date":date, "n1":mains[0], "n2":mains[1], "n3":mains[2], "n4":mains[3], "n5":mains[4], "sp":sp})
        time.sleep(sleep)
    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    df["date2"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date2").drop(columns=["date2"]).reset_index(drop=True)
    return df

def auto_update_if_needed():
    meta = {}
    if os.path.exists(META_JSON):
        meta = json.load(open(META_JSON, "r", encoding="utf-8"))
    today = datetime.now(VN_TZ).strftime("%Y-%m-%d")
    last = meta.get("last_update_date")
    if last == today and os.path.exists(DATA_CSV):
        return 0
    df_new = fetch_lotto8(pages=60)
    if os.path.exists(DATA_CSV):
        old = pd.read_csv(DATA_CSV)
        df = pd.concat([old, df_new], ignore_index=True).drop_duplicates().reset_index(drop=True)
    else:
        df = df_new
    df.to_csv(DATA_CSV, index=False, encoding="utf-8")
    meta["last_update_date"] = today
    meta["rows"] = len(df)
    json.dump(meta, open(META_JSON,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    return len(df)

# ---------------------- Predict ---------------------- #
def freq_series(df, cols=("n1","n2","n3","n4","n5")):
    main = df[list(cols)].values.flatten()
    f = pd.Series(main).value_counts().sort_index().reindex(range(1,36), fill_value=0)
    spf = df["sp"].value_counts().sort_index().reindex(range(1,13), fill_value=0)
    return f, spf

def normalize(v): return v / v.sum() if v.sum() > 0 else v

def weighted_pick_5(p_num, seed=20251106):
    rng = np.random.default_rng(seed)
    cand = p_num.index.to_numpy()
    probs = (p_num / p_num.sum()).to_numpy(dtype=float)
    return sorted(rng.choice(cand, size=5, replace=False, p=probs))

def dual_mode_predict(df_all, short_pages=20, long_pages=60, seed=20251106):
    n_short = min(len(df_all), short_pages*10)
    n_long  = min(len(df_all),  long_pages*10)
    df_short, df_long = df_all.tail(n_short), df_all.tail(n_long)
    f_s, spf_s = freq_series(df_short)
    f_l, spf_l = freq_series(df_long)
    w_num = normalize(0.6*normalize(f_s) + 0.4*normalize(f_l))
    w_sp  = normalize(0.6*normalize(spf_s) + 0.4*normalize(spf_l))
    rng_seed = seed
    tickets=[]
    for _ in range(5):
        mains = weighted_pick_5(w_num, seed=rng_seed)
        rng = np.random.default_rng(rng_seed)
        sp = int(rng.choice(w_sp.index.to_numpy(), p=w_sp.to_numpy()))
        tickets.append({"mains": mains, "sp": sp})
        rng_seed += 9
    return tickets

# ---------------------- Main ---------------------- #
with st.spinner("🔄 Đang cập nhật dữ liệu mới nhất..."):
    total_rows = auto_update_if_needed()

colA, colB = st.columns(2)
with colA:
    short_pages = st.slider("🟠 Cửa sổ NGẮN (bắt trend) – số trang", 10, 40, 20)
with colB:
    long_pages  = st.slider("🔵 Cửa sổ DÀI (bền vững) – số trang", 30, 80, 60)

if not os.path.exists(DATA_CSV):
    st.error("Chưa có dữ liệu. Hãy tải lại trang.")
    st.stop()

df_all = pd.read_csv(DATA_CSV)
tickets = dual_mode_predict(df_all, short_pages, long_pages)

st.subheader("🎫 5 Vé AI đề xuất (Hot + Stable)")
for i,t in enumerate(tickets, 1):
    st.markdown(f"""
    <div style="padding:10px 12px;border:1px solid #EEE;border-left:6px solid {VTL_RED};
                border-radius:12px;margin-bottom:8px;background:#fff;">
        <b>Vé #{i}:</b> <code>{t['mains']}</code> + <b>[ĐB {t['sp']:02d}]</b>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"<hr><div style='font-size:13px;color:#444;'>Auto update hàng ngày • Dữ liệu: {len(df_all)} kỳ</div>", unsafe_allow_html=True)

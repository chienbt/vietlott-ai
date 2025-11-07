# 🎯 AI Vietlott 5/35 – v2_auto_full (Auto update + Refresh button + Vietlott UI)
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

st.set_page_config(page_title="AI Vietlott 5/35 – Dual-Mode (Auto)",
                   page_icon="🎯", layout="wide")

# ---------------------- UI header ---------------------- #
VTL_RED = "#E10600"
VTL_YELLOW = "#F5C400"

st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;padding:14px 10px;
            border-radius:14px;background:linear-gradient(90deg,{VTL_RED}0D,#fff 40%);">
  <img src="https://raw.githubusercontent.com/chienbt/vietlott-ai/main/logo_vietlott.svg" height="48"/>
  <div>
    <div style="font-weight:800;font-size:28px;line-height:1;color:#111;">
      AI Vietlott 5/35 – Dual-Mode Predictor (Hot + Stable)
    </div>
    <div style="color:#444;font-size:14px;margin-top:4px">
      Phân tích song song cửa sổ <b>NGẮN</b> (bắt trend) + <b>DÀI</b> (xác suất bền).
      Tự động cập nhật theo ngày & cho phép <b>🔄 làm mới vé ngay</b>.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------- Fetch & Auto update ---------------------- #
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
            date = f"{int(m_date.group(1)):04d}-{int(m_date.group(2)):02d}-{int(m_date.group(3)):02d}" if m_date else ""
            rows.append({"date":date, "n1":mains[0], "n2":mains[1], "n3":mains[2], "n4":mains[3], "n5":mains[4], "sp":sp})
        time.sleep(sleep)
    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    try:
        df["date2"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date2").drop(columns=["date2"]).reset_index(drop=True)
    except Exception:
        pass
    return df

def load_meta():
    if os.path.exists(META_JSON):
        try: return json.load(open(META_JSON,"r",encoding="utf-8"))
        except Exception: return {}
    return {}
def save_meta(meta:dict):
    with open(META_JSON,"w",encoding="utf-8") as f: json.dump(meta,f,ensure_ascii=False,indent=2)

def auto_update_if_needed():
    meta = load_meta()
    today = datetime.now(VN_TZ).strftime("%Y-%m-%d")
    if meta.get("last_update_date")==today and os.path.exists(DATA_CSV):
        return
    df_new = fetch_lotto8(pages=60)         # đủ cho cửa sổ dài
    if os.path.exists(DATA_CSV):
        try:
            old = pd.read_csv(DATA_CSV)
            df = pd.concat([old, df_new], ignore_index=True).drop_duplicates().reset_index(drop=True)
        except Exception:
            df = df_new
    else:
        df = df_new
    df.to_csv(DATA_CSV, index=False, encoding="utf-8")
    meta["last_update_date"] = today
    meta["rows"] = int(len(df))
    save_meta(meta)

with st.spinner("🔄 Đang kiểm tra & cập nhật dữ liệu trong ngày..."):
    auto_update_if_needed()

# ---------------------- Analytics ---------------------- #
def freq_series(df, cols=("n1","n2","n3","n4","n5")):
    main = df[list(cols)].values.flatten()
    f = pd.Series(main).value_counts().sort_index().reindex(range(1,36), fill_value=0)
    spf = df["sp"].value_counts().sort_index().reindex(range(1,13), fill_value=0)
    return f, spf

def normalize(v):
    v = v.astype(float); s = v.sum()
    return (v/s) if s>0 else pd.Series([1/len(v)]*len(v), index=v.index)

def weighted_pick_5(p_num: pd.Series, seed=20251106):
    rng = np.random.default_rng(seed)
    cand = p_num.index.to_numpy()
    probs = (p_num / p_num.sum()).to_numpy(dtype=float)
    return sorted(int(x) for x in rng.choice(cand, size=5, replace=False, p=probs))

def dual_mode_predict(df_all: pd.DataFrame, short_pages=20, long_pages=60, seed=20251106):
    n_short = min(len(df_all), short_pages*10)
    n_long  = min(len(df_all),  long_pages*10)
    df_short = df_all.tail(n_short) if n_short>0 else df_all
    df_long  = df_all.tail(n_long)  if n_long>0  else df_all
    f_s, spf_s = freq_series(df_short)
    f_l, spf_l = freq_series(df_long)
    w_num = normalize(0.6*normalize(f_s) + 0.4*normalize(f_l))
    w_sp  = normalize(0.6*normalize(spf_s) + 0.4*normalize(spf_l))
    tickets=[]
    rng_seed = seed
    for _ in range(5):
        mains = weighted_pick_5(w_num, seed=rng_seed)
        rng = np.random.default_rng(rng_seed)
        sp = int(rng.choice(w_sp.index.to_numpy(), p=w_sp.to_numpy()))
        tickets.append({"mains": mains, "sp": sp})
        rng_seed += 11
    return tickets, (f_s, f_l), (spf_s, spf_l)

def top10_best_next_hit(df_all: pd.DataFrame):
    cols = ["n1","n2","n3","n4","n5"]
    rows=[]
    for i in range(len(df_all)-1):
        cur = tuple(sorted(int(x) for x in df_all.loc[i, cols].tolist()))
        nxt = set(int(x) for x in df_all.loc[i+1, cols].tolist())
        sp_hit = int(int(df_all.loc[i, "sp"]) == int(df_all.loc[i+1, "sp"]))
        hit = len(set(cur) & nxt)
        score = hit*100 + (50 if sp_hit else 0)  # 5+ĐB > 5 > 4+ĐB ...
        rows.append((cur, score, sp_hit))
    if not rows: return pd.DataFrame(columns=["combo","Loại trúng","count"])
    hist = pd.DataFrame(rows, columns=["combo","score","sp"])
    agg = (hist.value_counts(["combo","score","sp"])
           .reset_index(name="count")
           .sort_values(["score","count"], ascending=[False,False])
           .head(10))
    def label(score, sp):
        hit = score//100
        m={ (5,1):"5 số + ĐB",(5,0):"5 số",(4,1):"4 số + ĐB",(4,0):"4 số",
            (3,1):"3 số + ĐB",(3,0):"3 số",(2,1):"2 số + ĐB",(2,0):"2 số",
            (1,1):"1 số + ĐB",(1,0):"1 số",(0,1):"Đặc biệt",(0,0):"Không trúng"}
        return m.get((hit,sp),"Khác")
    agg["Loại trúng"] = [label(r.score, r.sp) for r in agg.itertuples()]
    agg["combo"] = agg["combo"].apply(lambda t: " ".join(f"{int(x):02d}" for x in t))
    return agg[["combo","Loại trúng","count"]]

# ---------------------- Controls ---------------------- #
colA, colB, colC = st.columns([1,1,1])
with colA:
    short_pages = st.slider("🟠 Cửa sổ NGẮN (bắt trend) – số trang", 10, 40, 20)
with colB:
    long_pages  = st.slider("🔵 Cửa sổ DÀI (bền vững) – số trang", 30, 80, 60)
with colC:
    do_refresh = st.button("🔄 Làm mới & sinh vé khác")

# ---------------------- Load & Predict ---------------------- #
if not os.path.exists(DATA_CSV):
    st.error("Chưa có dữ liệu. Hãy tải lại trang.")
    st.stop()
df_all = pd.read_csv(DATA_CSV)

# seed cố định (ổn định) hoặc đổi khi bấm nút
base_seed = int(short_pages * 1000 + long_pages)
tickets, (f_s, f_l), (spf_s, spf_l) = dual_mode_predict(df_all, short_pages, long_pages, seed=base_seed)

# ---------------------- Render ---------------------- #
fmt = lambda arr: " ".join(f"{int(x):02d}" for x in arr)

st.subheader("🎫 5 Vé AI đề xuất (Hot + Stable)")
for i,t in enumerate(tickets, 1):
    st.markdown(
        f"<div style='padding:10px 12px;border:1px solid #EEE;border-left:6px solid {VTL_RED};"
        f"border-radius:12px;margin-bottom:8px;background:#fff;'>"
        f"<b>Vé #{i}:</b> <code>{fmt(t['mains'])}</code>  +  <b>[ĐB {int(t['sp']):02d}]</b>"
        f"</div>", unsafe_allow_html=True
    )

st.divider()
st.subheader("🏆 Top 10 tổ hợp trúng cao nhất (ưu tiên: 5+ĐB → 5 → 4+ĐB → …)")
top10 = top10_best_next_hit(df_all)
st.dataframe(top10, use_container_width=True)

st.divider()
st.subheader("📈 Tần suất số chính & đặc biệt – So sánh NGẮN vs DÀI")
c1,c2 = st.columns(2)
with c1:
    st.markdown("**Số chính – NGẮN hạn**"); st.bar_chart(f_s)
    st.markdown("**Số chính – DÀI hạn**");  st.bar_chart(f_l)
with c2:
    st.markdown("**Đặc biệt – NGẮN hạn**"); st.bar_chart(spf_s)
    st.markdown("**Đặc biệt – DÀI hạn**");  st.bar_chart(spf_l)

meta = load_meta()
st.markdown(
    f"<div style='margin-top:12px;padding:8px 12px;border-radius:10px;background:#FFF7E0;border:1px solid #FFE08A;'>"
    f"<b>Auto-update:</b> Cập nhật lần cuối: <b>{meta.get('last_update_date','—')}</b> (giờ VN). "
    f"Lịch sử: <b>{meta.get('rows','—')}</b> kỳ. "
    f"{'🔁 Đã làm mới vé' if do_refresh else ''}"
    f"</div>", unsafe_allow_html=True
)

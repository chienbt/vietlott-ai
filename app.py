# 🎯 AI Vietlott 5/35 – v2 (Dual-Mode Predictor)
# 👉 Chạy: pip install streamlit pandas numpy requests beautifulsoup4
#         streamlit run app.py

import streamlit as st
import requests, re, time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import itertools

st.set_page_config(page_title="AI Vietlott 5/35 – Dual-Mode", layout="wide", page_icon="🎯")
st.title("🎯 AI Vietlott 5/35 – Dual-Mode Predictor (Hot + Stable)")
st.caption("Phân tích song song: cửa sổ **ngắn hạn** (bắt trend) + **dài hạn** (xác suất bền). Hợp nhất → 5 vé đề xuất. Đồng thời xếp hạng Top 10 tổ hợp trúng cao nhất lịch sử (5+ĐB → 5 → 4+ĐB → ...).")

# ----------------------------- Crawl dữ liệu ----------------------------- #
def fetch_lotto8(pages=20, sleep=0.15):
    base = "https://www.lotto-8.com/Vietnam/listltoVM35.asp"
    all_rows=[]
    for p in range(1, pages+1):
        r = requests.get(base, params={"page":p}, timeout=12)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tr in soup.select("table tr"):
            txt = tr.get_text(" ", strip=True)
            nums = [int(x) for x in re.findall(r'\b([0-2]?\d|3[0-5])\b', txt)]
            if len(nums) < 6:
                continue
            # tách 5 số chính + 1 đặc biệt
            mains = [n for n in nums if 1 <= n <= 35]
            sps   = [n for n in nums if 1 <= n <= 12]
            mains = sorted(list(dict.fromkeys(mains)))[:5]
            if len(mains) != 5 or not sps:
                continue
            sp = sps[-1]
            m_date = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', txt)
            date = ""
            if m_date:
                y, mo, d = map(int, m_date.groups())
                date = f"{y:04d}-{mo:02d}-{d:02d}"
            row = {"date":date, "n1":mains[0], "n2":mains[1], "n3":mains[2], "n4":mains[3], "n5":mains[4], "sp":sp}
            all_rows.append(row)
        time.sleep(sleep)
    df = pd.DataFrame(all_rows).drop_duplicates().reset_index(drop=True)
    # Chuẩn hoá & sắp xếp theo ngày (nếu có)
    try:
        df["date2"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date2").drop(columns=["date2"]).reset_index(drop=True)
    except Exception:
        pass
    return df

# ----------------------------- Tiện ích phân tích ----------------------------- #
def freq_series(df, cols_main=("n1","n2","n3","n4","n5")):
    main = df[list(cols_main)].values.flatten()
    f = pd.Series(main).value_counts().sort_index()
    f = f.reindex(range(1,36), fill_value=0)
    spf = df["sp"].value_counts().sort_index().reindex(range(1,13), fill_value=0)
    return f, spf

def normalize(v):
    v = v.astype(float)
    s = v.sum()
    if s <= 0:
        return pd.Series([1/len(v)]*len(v), index=v.index)
    return v / s

def weighted_pick_5(p_num: pd.Series, seed=20251106):
    """Bốc 5 số theo phân bố p_num (không lặp)."""
    rng = np.random.default_rng(seed)
    candidates = p_num.index.to_numpy()
    probs = p_num.to_numpy(dtype=float)
    probs = probs / probs.sum()
    picks = sorted(rng.choice(candidates, size=5, replace=False, p=probs))
    return picks

def label_rank(hit_main:int, hit_sp:bool):
    # Trả về (score, label) để sort theo ưu tiên 5+ĐB > 5 > 4+ĐB > ...
    # Dùng score  = hit_main*100 + (50 nếu trúng ĐB)
    score = hit_main*100 + (50 if hit_sp else 0)
    label_map = {
        (5, True): "5 số + Đặc biệt",
        (5, False): "5 số",
        (4, True): "4 số + Đặc biệt",
        (4, False): "4 số",
        (3, True): "3 số + Đặc biệt",
        (3, False): "3 số",
        (2, True): "2 số + Đặc biệt",
        (2, False): "2 số",
        (1, True): "1 số + Đặc biệt",
        (1, False): "1 số",
        (0, True): "Đặc biệt",
        (0, False): "Không trúng"
    }
    return score, label_map.get((hit_main, hit_sp), "Khác")

def top10_best_next_hit(df_all: pd.DataFrame):
    """Quét lịch sử: combo kỳ t so với kỳ t+1; đếm và xếp hạng theo ưu tiên."""
    cols = ["n1","n2","n3","n4","n5"]
    rows=[]
    for i in range(len(df_all)-1):
        cur = tuple(sorted(int(x) for x in df_all.loc[i, cols].tolist()))
        nxt = set(int(x) for x in df_all.loc[i+1, cols].tolist())
        sp_hit = int(int(df_all.loc[i, "sp"]) == int(df_all.loc[i+1, "sp"]))
        hit = len(set(cur) & nxt)
        score, label = label_rank(hit, bool(sp_hit))
        rows.append((cur, score, label))
    if not rows:
        return pd.DataFrame(columns=["combo","Loại trúng","count"])
    hist = pd.DataFrame(rows, columns=["combo","score","label"])
    agg = (hist.value_counts(["combo","score","label"])
           .reset_index(name="count")
           .sort_values(["score","count"], ascending=[False, False])
           .head(10))
    # Hiển thị combo dạng 01 02 ...
    agg["combo"] = agg["combo"].apply(lambda t: " ".join(f"{x:02d}" for x in t))
    agg = agg.rename(columns={"label":"Loại trúng"})
    return agg[["combo","Loại trúng","count"]]

# ----------------------------- Dual-Mode Predictor ----------------------------- #
def dual_mode_predict(df_all: pd.DataFrame, short_pages=20, long_pages=60, seed=20251106):
    """Tạo 5 vé dựa trên cửa sổ ngắn & dài hạn, hợp nhất trọng số."""
    # Cắt hai cửa sổ (theo trang: 1 trang ≈ 10 kỳ)
    n_short = min(len(df_all), short_pages*10)
    n_long  = min(len(df_all),  long_pages*10)
    df_short = df_all.tail(n_short) if n_short>0 else df_all.copy()
    df_long  = df_all.tail(n_long)  if n_long>0  else df_all.copy()

    # Tần suất (số chính & đặc biệt)
    f_s, spf_s = freq_series(df_short)
    f_l, spf_l = freq_series(df_long)

    # Trọng số kết hợp (ưu tiên trend gần đây nhưng vẫn giữ bền vững)
    w_num = normalize(0.6*normalize(f_s) + 0.4*normalize(f_l))
    w_sp  = normalize(0.6*normalize(spf_s) + 0.4*normalize(spf_l))

    # Sinh 5 vé
    rng_seed = seed
    tickets=[]
    for _ in range(5):
        mains = weighted_pick_5(w_num, seed=rng_seed)
        rng = np.random.default_rng(rng_seed)
        sp = int(rng.choice(w_sp.index.to_numpy(), p=w_sp.to_numpy()))
        tickets.append({"mains": mains, "sp": sp})
        rng_seed += 7  # đổi seed nhẹ để đa dạng

    return tickets, (w_num, w_sp), (f_s, f_l), (spf_s, spf_l)

# ----------------------------- Giao diện ----------------------------- #
colA, colB, colC = st.columns([1.1,1,1.1])
with colA:
    short_pages = st.slider("🟠 Cửa sổ NGẮN (bắt trend) – số trang", 10, 40, 20, help="~10 kỳ/1 trang. 20 trang ≈ 200 kỳ.")
with colC:
    long_pages  = st.slider("🔵 Cửa sổ DÀI (bền vững) – số trang", 30, 80, 60, help="Cân bằng ổn định dài hạn. 60 trang ≈ 600 kỳ.")

if st.button("🚀 Phân tích & Sinh 5 vé AI (Dual-Mode)"):
    with st.spinner("Đang lấy dữ liệu & phân tích hai cửa sổ..."):
        # Lấy dữ liệu theo cửa sổ dài (để đủ cho cả ngắn)
        max_pages = max(short_pages, long_pages)
        df_all = fetch_lotto8(pages=max_pages)
        tickets, (w_num, w_sp), (f_s, f_l), (spf_s, spf_l) = dual_mode_predict(
            df_all, short_pages=short_pages, long_pages=long_pages
        )
        top10 = top10_best_next_hit(df_all)

    st.subheader("🎫 5 Vé AI đề xuất (giao điểm Hot + Bền)")
    for i,t in enumerate(tickets, 1):
        st.write(f"**Vé #{i}:** {t['mains']}  +  [ĐB {t['sp']:02d}]")

    st.divider()
    st.subheader("🏆 Top 10 tổ hợp trúng cao nhất (xếp theo: 5+ĐB → 5 → 4+ĐB → ...)")
    if len(top10)==0:
        st.info("Chưa đủ dữ liệu để thống kê.")
    else:
        st.dataframe(top10, use_container_width=True)

    st.divider()
    st.subheader("📈 Tần suất số chính & đặc biệt – So sánh NGẮN vs DÀI")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Số chính – NGẮN hạn**")
        st.bar_chart(f_s)
        st.markdown("**Số chính – DÀI hạn**")
        st.bar_chart(f_l)
    with c2:
        st.markdown("**Đặc biệt – NGẮN hạn**")
        st.bar_chart(spf_s)
        st.markdown("**Đặc biệt – DÀI hạn**")
        st.bar_chart(spf_l)

    st.success("✅ Hoàn tất! Bố có thể đổi 2 thanh trượt rồi bấm lại để so sánh phương án.")
else:
    st.info("👆 Chọn 2 cửa sổ dữ liệu (NGẮN & DÀI), sau đó nhấn nút 🚀 để bắt đầu phân tích song song.")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# --- Load model, fitur, dan encoder ---
with open("models/people_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_cols = pickle.load(f)

with open("models/label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# --- Konfigurasi halaman ---
st.set_page_config(page_title="People Analytics", page_icon="🧑‍💼", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("📁 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["📊 Dashboard Insight", "🔮 Prediksi Karyawan Baru"])

# --- Fungsi load data ---
@st.cache_data
def load_data():
    df = pd.read_excel("data/EmployeeSurvey.xlsx", engine="openpyxl")
    df = df[df["job_satisfaction"].between(1, 5)]  # Filter nilai tidak valid
    return df

df = load_data()

# ============================ #
# 📊 DASHBOARD PANEL
# ============================ #
if page == "📊 Dashboard Insight":
    st.title("📊 People Analytics Dashboard")
    st.markdown("Analisis data kepuasan kerja karyawan berdasarkan faktor internal dan keseimbangan hidup.")
    st.markdown("---")
    # Ringkasan cepat
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Karyawan", len(df))
    col2.metric("Rata-rata Usia", f"{df['age'].mean():.1f} tahun")
    col3.metric("Rata-rata Jam Pelatihan", f"{df['training_hours_per_year'].mean():.1f} jam/tahun")
    st.markdown("---")
    # --- Filter Panel ---
    st.sidebar.header("🎛 Filter Data")
    dept_filter = st.sidebar.multiselect("Departemen", options=df["dept"].unique(), default=list(df["dept"].unique()))
    age_min, age_max = st.sidebar.slider("Rentang Usia", min_value=int(df["age"].min()), max_value=int(df["age"].max()), value=(25, 45))

    # Terapkan filter
    filtered_df = df[(df["dept"].isin(dept_filter)) & (df["age"].between(age_min, age_max))]

    # Visualisasi 1: Distribusi job_satisfaction
    st.subheader("📈 Distribusi Kepuasan Kerja")
    fig1 = px.histogram(
        filtered_df,
        x="job_satisfaction",
        nbins=5,
        title="Distribusi Kepuasan Kerja (Skala 1–5)",
        labels={"job_satisfaction": "Tingkat Kepuasan"},
        color_discrete_sequence=["#636EFA"]
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Visualisasi 2: Rata-rata stres per departemen
    st.subheader("🔥 Rata-rata Tingkat Stres per Departemen")
    fig2 = px.bar(
        filtered_df.groupby("dept")["stress"].mean().reset_index(),
        x="dept",
        y="stress",
        color="stress",
        color_continuous_scale="Reds",
        title="Stres Rata-rata"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Visualisasi 3: Aktivitas Fisik vs Kepuasan
    st.subheader("⚽ Aktivitas Fisik vs Kepuasan Kerja")
    fig3 = px.scatter(
        filtered_df,
        x="physical_activity_hours",
        y="job_satisfaction",
        color="dept",
        size="stress",
        hover_data=["age", "wlb", "work_env"],
        title="Korelasi Aktivitas Fisik & Kepuasan"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Visualisasi 4: Workload vs Job Satisfaction
    st.subheader("📉 Beban Kerja vs Kepuasan")
    fig4 = px.box(filtered_df, x="job_satisfaction", y="workload", color="job_satisfaction",
                  title="Distribusi Beban Kerja per Tingkat Kepuasan")
    st.plotly_chart(fig4, use_container_width=True)

    # Visualisasi 5: Sleep Hours vs Job Satisfaction
    st.subheader("😴 Tidur vs Kepuasan")
    fig5 = px.violin(filtered_df, y="sleep_hours", x="job_satisfaction", color="job_satisfaction",
                     box=True, points="all", title="Jam Tidur vs Kepuasan Kerja")
    st.plotly_chart(fig5, use_container_width=True)

    # Visualisasi 6: Korelasi Numerik (bar chart)
    st.subheader("🧠 Korelasi Fitur terhadap Job Satisfaction")
    correlation = filtered_df.select_dtypes(include=np.number).corr()["job_satisfaction"].drop("job_satisfaction")
    top_corr = correlation.sort_values(key=abs, ascending=False).head(10)
    fig6 = px.bar(
        top_corr.reset_index(),
        x="index",
        y="job_satisfaction",
        color="job_satisfaction",
        color_continuous_scale="RdBu",
        title="Top 10 Korelasi Fitur dengan Job Satisfaction",
        labels={"index": "Fitur", "job_satisfaction": "Korelasi"}
    )
    st.plotly_chart(fig6, use_container_width=True)

    # Insight
    with st.expander("📌 Insight & Rekomendasi"):
        st.markdown("""
        - **Stres tinggi** ditemukan di beberapa departemen — perlu program manajemen stres atau workload balance.
        - **Aktivitas fisik** berkorelasi positif terhadap kepuasan.
        - **Tidur** yang cukup dapat meningkatkan kepuasan.
        - Karyawan **usia 30–40 tahun** tampak memiliki kepuasan kerja lebih stabil.
        """)

# ============================ #
# 🔮 PREDIKSI PANEL
# ============================ #
elif page == "🔮 Prediksi Karyawan Baru":
    st.title("🔮 Prediksi Tingkat Kepuasan Karyawan Baru")
    st.markdown("Isi form berikut untuk memprediksi tingkat kepuasan kerja seorang karyawan.")

    with st.form("prediction_form"):
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        age = st.slider("Usia", 18, 60, 30)
        marital_status = st.selectbox("Status Pernikahan", ["Single", "Married"])
        job_level = st.selectbox("Level Jabatan", df["job_level"].unique())
        experience = st.slider("Pengalaman Kerja (tahun)", 0, 40, 5)
        dept = st.selectbox("Departemen", df["dept"].unique())
        emp_type = st.selectbox("Tipe Karyawan", df["emp_type"].unique())
        wlb = st.slider("Work-Life Balance (1-5)", 1, 5, 3)
        work_env = st.slider("Kualitas Lingkungan Kerja (1-5)", 1, 5, 3)
        physical_activity_hours = st.slider("Jam Aktivitas Fisik / Minggu", 0.0, 15.0, 3.0)
        workload = st.slider("Tingkat Beban Kerja (1-5)", 1, 5, 3)
        stress = st.slider("Tingkat Stres (1-5)", 1, 5, 3)
        sleep_hours = st.slider("Jam Tidur / Hari", 0.0, 12.0, 7.0)
        commute_mode = st.selectbox("Moda Transportasi", df["commute_mode"].unique())
        commute_distance = st.slider("Jarak ke Kantor (km)", 0, 100, 10)
        num_companies = st.slider("Jumlah Perusahaan Sebelumnya", 0, 10, 2)
        team_size = st.slider("Ukuran Tim Saat Ini", 1, 100, 10)
        num_reports = st.slider("Jumlah Laporan Langsung", 0, 10, 0)
        edu_level = st.selectbox("Pendidikan Terakhir", df["edu_level"].unique())
        have_ot = st.selectbox("Sering Lembur?", [True, False])
        training_hours = st.slider("Jam Pelatihan / Tahun", 0.0, 100.0, 20.0)

        submitted = st.form_submit_button("🔮 Prediksi")

        if submitted:
            input_dict = {
                "gender": gender,
                "age": age,
                "marital_status": marital_status,
                "job_level": job_level,
                "experience": experience,
                "dept": dept,
                "emp_type": emp_type,
                "wlb": wlb,
                "work_env": work_env,
                "physical_activity_hours": physical_activity_hours,
                "workload": workload,
                "stress": stress,
                "sleep_hours": sleep_hours,
                "commute_mode": commute_mode,
                "commute_distance": commute_distance,
                "num_companies": num_companies,
                "team_size": team_size,
                "num_reports": num_reports,
                "edu_level": edu_level,
                "have_ot": have_ot,
                "training_hours_per_year": training_hours,
            }

            input_df = pd.DataFrame([input_dict])

            # Encoding
            for col in input_df.columns:
                if col in encoders:
                    input_df[col] = encoders[col].transform(input_df[col])

            for col in feature_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_cols]

            # Prediksi
            prediction = model.predict(input_df)[0]
            st.success(f"🎯 Prediksi Tingkat Kepuasan Kerja: **{prediction}** (Skala 1–5)")

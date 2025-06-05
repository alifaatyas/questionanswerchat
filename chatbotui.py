import streamlit as st
import faiss
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
import openai

# --- Konfigurasi Awal ---
st.set_page_config(
    page_title="Asisten Hukum",
    layout="wide",  # Menggunakan layout lebar untuk lebih banyak ruang
    initial_sidebar_state="expanded",
    page_icon="⚖️" # Menambahkan ikon halaman
)
# --- Definisi Avatar Kustom ---
USER_ICON_COLOR = "#64B5F6"  # Biru muda untuk pengguna
AI_ICON_COLOR = "#4DD0E1"    # Cyan cerah untuk AI

USER_AVATAR_SVG = f"""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{USER_ICON_COLOR}" width="28px" height="28px">
  <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
  <path d="M0 0h24v24H0z" fill="none"/>
</svg>
"""

AI_AVATAR_SVG = f"""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{AI_ICON_COLOR}" width="28px" height="28px">><title>robot</title><path d="M12,2A2,2 0 0,1 14,4C14,4.74 13.6,5.39 13,5.73V7H14A7,7 0 0,1 21,14H22A1,1 0 0,1 23,15V18A1,1 0 0,1 22,19H21V20A2,2 0 0,1 19,22H5A2,2 0 0,1 3,20V19H2A1,1 0 0,1 1,18V15A1,1 0 0,1 2,14H3A7,7 0 0,1 10,7H11V5.73C10.4,5.39 10,4.74 10,4A2,2 0 0,1 12,2M7.5,13A2.5,2.5 0 0,0 5,15.5A2.5,2.5 0 0,0 7.5,18A2.5,2.5 0 0,0 10,15.5A2.5,2.5 0 0,0 7.5,13M16.5,13A2.5,2.5 0 0,0 14,15.5A2.5,2.5 0 0,0 16.5,18A2.5,2.5 0 0,0 19,15.5A2.5,2.5 0 0,0 16.5,13Z" /></svg>
"""

# --- Sidebar Informasi tentang Legal Drafting ---
with st.sidebar:
    st.markdown("## ⚖️ Legal Drafting")
    st.markdown("""
    Selamat datang di **Asisten Hukum**! Aplikasi ini dirancang untuk membantu Anda dalam proses penyusunan dokumen hukum yang kompleks. Temukan pasal-pasal relevan dengan mudah dan cepat.

    📝 **Legal drafting** adalah seni dan ilmu penyusunan dokumen hukum seperti kontrak, undang-undang, atau perjanjian hukum yang efektif. Dokumen yang baik haruslah:

    - 🎯 **Jelas & Tepat:** Bahasa yang mudah dipahami dan tidak ambigu.
    - 📜 **Sah & Mengikat:** Sesuai dengan kerangka hukum yang berlaku.
    - 🧠 **Sistematis & Logis:** Struktur yang runtut dan argumentasi yang kokoh.

    Manfaatkan fitur pencarian cerdas kami untuk mendukung pekerjaan Anda!
    """)
    st.markdown("---")
    # Anda bisa menambahkan gambar yang relevan di sini jika mau
    # st.image("path/to/your/image.png", use_column_width=True)
    # Contoh menggunakan emoji besar sebagai placeholder visual
    st.markdown("<p style='text-align:center; font-size: 4em;'>🏛️</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 👨‍💻 Tim Capstone")
    st.markdown("""
    - 🌟 Siti Nurafifa Zainul Karima
    - 🚀 Muhammad Mirza Zulhilmi
    - ✨ Muhammad Aqil Ghazali Anhein
    - 💡 Alifa Tyas Khairunisa
    """)
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #A0A0A0; font-size: 0.8em;'>Versi Aplikasi 1.0 | Juni 2025</p>", unsafe_allow_html=True)


# --- Styling CSS agar tampilan lebih menarik ---
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

        body {{
            background-color: #0E1117;
            color: #FAFAFA;
            font-family: 'Roboto', sans-serif;
        }}

        /* Styling Sidebar */
        [data-testid="stSidebar"] {{
            background-color: #14181F !important;
            border-right: 1px solid #2A2D34;
        }}
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {{
            color: #BB86FC; /* Changed from #00A79D (green) to electric purple */
            font-weight: 500;
        }}
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown li {{
            color: #C5C6C7;
            font-size: 0.9rem;
            line-height: 1.6;
        }}
        [data-testid="stSidebar"] .stMarkdown strong {{
            color: #BB86FC; /* Changed from #00C2B3 (green) to electric purple */
        }}

        /* Area Konten Utama */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }}

        .app-title {{
            text-align: center;
            font-size: 2.6em;
            font-weight: 700;
            color: #BB86FC; /* Changed from #00A79D (green) to electric purple */
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }}

        .app-subtitle {{
            text-align: center;
            font-size: 1.15em;
            color: #A0A0A0;
            margin-bottom: 40px;
            font-weight: 300;
        }}

        /* Input Chat */
        .stChatInputContainer > div > input {{
            background-color: #1A1D24;
            color: #FAFAFA;
            border: 1px solid #3a3f4b;
            border-radius: 8px;
            padding: 12px 18px;
            font-size: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stChatInputContainer > div > input::placeholder {{
            color: #707070;
        }}
        .stChatInputContainer button {{
            background-color: #BB86FC !important; /* Changed from #00A79D (green) to electric purple */
            border: none !important;
            border-radius: 8px !important;
            padding: 0px 15px !important;
            transition: background-color 0.2s ease;
        }}
        .stChatInputContainer button:hover {{
            background-color: #9A5CE8 !important; /* Slightly darker purple on hover */
        }}
        .stChatInputContainer button svg {{
             fill: #FFFFFF !important;
        }}

        /* Bubble Chat */
        .stChatMessage {{
            border-radius: 12px;
            padding: 16px 22px; /* Sedikit penyesuaian padding jika avatar lebih besar */
            margin-bottom: 12px;
            box-shadow: 0 3px 6px rgba(0,0,0,0.2);
            border: none;
            gap: 12px; /* Menambah jarak antara avatar dan bubble konten */
        }}

        .stChatMessage [data-testid="chatAvatarIcon-custom"] {{ /* Container untuk avatar kustom */
            display: flex;
            align-items: center;
            justify-content: center;
            width: 32px;  /* Sesuaikan dengan ukuran SVG atau keinginan Anda */
            height: 32px; /* Sesuaikan dengan ukuran SVG atau keinginan Anda */
            border-radius: 50%; /* Membuat background (jika ada) menjadi lingkaran */
            /* background-color: rgba(255, 255, 255, 0.08); */ /* Contoh background halus */
            /* box-shadow: 0 1px 2px rgba(0,0,0,0.1); */ /* Contoh bayangan halus untuk avatar */
        }}
        .stChatMessage [data-testid="chatAvatarIcon-custom"] svg {{
            /* Ukuran SVG diatur dalam string SVG, tapi bisa di-override di sini jika perlu */
            /* width: 28px; */
            /* height: 28px; */
            position: relative;
            top: 0px; /* Penyesuaian posisi vertikal jika ikon tidak pas */
        }}
        /* --- AKHIR STYLING BARU UNTUK AVATAR KUSTOM --- */

        /* Bubble Chat Pengguna */
        [data-testid="chat-message-container-user"] > [data-testid="stChatMessageContent"] {{
            background-color: #007AFF; /* Kept blue, it contrasts well with purple and is common for user bubbles */
        }}
        [data-testid="chat-message-container-user"] .stMarkdown p {{
            color: #FFFFFF !important;
            font-size: 1rem;
            line-height: 1.6;
        }}

        /* Bubble Chat Asisten & Kotak Jawaban */
        [data-testid="chat-message-container-assistant"] > [data-testid="stChatMessageContent"] {{
            background-color: #2C2F36;
        }}

        .response-box {{
            background-color: transparent;
            border: none;
            padding: 0;
            margin-top: 0;
            color: #EAEAEA;
        }}
        .response-box h4 {{
            color: #BB86FC; /* Changed from #00A79D (green) to electric purple */
            font-size: 1.1em;
            margin-bottom: 10px;
            font-weight: 500;
            border-bottom: 1px solid #3A3D44;
            padding-bottom: 8px;
        }}
        .response-box p, .response-box div, .response-box ul, .response-box li {{
            font-size: 0.98rem;
            line-height: 1.65;
            color: #D5D5D5;
        }}
         .response-box strong {{
            color: #D9BBFF; /* Lighter shade of purple for strong text in response box */
            font-weight: 500;
        }}
        .response-box br {{
            display: block;
            margin-bottom: 0.6em;
            content: "";
        }}

        /* Styling untuk pesan error/warning/info (sudah ada) */
        .stAlert {{
            border-radius: 8px;
            font-size: 0.95rem;
            border-left-width: 5px !important;
            padding: 1rem;
        }}
        .stAlert p, .stAlert li {{
            color: #111111 !important;
        }}
        [data-testid="stAlert"][data-baseweb="notification"][kind="error"] {{
            border-left-color: #FF4B4B !important;
            background-color: #FFE0E0 !important;
        }}
        [data-testid="stAlert"][data-baseweb="notification"][kind="warning"] {{
            border-left-color: #FFC400 !important;
            background-color: #FFF3CD !important;
        }}
        [data-testid="stAlert"][data-baseweb="notification"][kind="info"] {{
            border-left-color: #BB86FC !important; /* Changed from #00A79D (green) to electric purple */
            background-color: #EDDFFC !important; /* Adjusted background for info alert to match new accent */
        }}

        /* Warna Spinner (MODIFIKASI) */
        .stSpinner > div > div {{ /* Ini menargetkan ikon spinner grafisnya */
            border-top-color: #BB86FC !important; /* Changed from #00A79D (green) to electric purple */
            border-right-color: transparent !important;
            border-bottom-color: transparent !important;
            border-left-color: transparent !important;
            width: 24px !important; /* Ukuran ikon spinner disesuaikan */
            height: 24px !important; /* Ukuran ikon spinner disesuaikan */
        }}

        [data-testid="stSpinner"] {{ /* Container utama spinner */
            display: flex;
            align-items: center; /* Menyejajarkan ikon dan teks spinner secara vertikal */
            justify-content: flex-start; /* Memulai dari kiri */
            gap: 10px; /* Jarak antara ikon spinner dan teksnya */
        }}

        .stSpinner p {{ /* Teks di dalam spinner */
            color: #A0A0A0;
            font-size: 0.9em;
            line-height: 1.5; /* Menyesuaikan line-height jika perlu */
            white-space: normal; /* Memungkinkan teks untuk wrap jika kontainer tidak cukup */
            /* Jika Anda ingin memaksa satu baris, gunakan: white-space: nowrap; */
            margin: 0; /* Menghilangkan margin default dari paragraf */
        }}

    </style>
""", unsafe_allow_html=True)

# Menggunakan class kustom untuk judul dan subjudul agar CSS bisa diterapkan
st.markdown('<div class="app-title">💬 Asisten Hukum: Tanya Jawab Pasal</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Sampaikan pertanyaan hukum Anda, kami bantu telusuri isi pasal yang relevan dengan dukungan AI canggih.</div>', unsafe_allow_html=True)

openai.api_key = st.secrets["OPENAI_API_KEY"] #secret key OpenAI Anda di sini

# Load FAISS dan metadata (gunakan cache agar cepat)
@st.cache_resource
def load_faiss_and_metadata():
    # Ensure file paths are correct for your environment
    try:
        index = faiss.read_index("pasal_index.faiss")
        metadata = pd.read_json("pasal_metadata.json")
        return index, metadata
    except FileNotFoundError as e:
        st.error(f"Error loading FAISS/metadata file: {e}. Please check the file paths.")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading FAISS/metadata: {e}")
        return None, None

index, metadata = load_faiss_and_metadata()

# Koneksi Neo4j (cache untuk performa)
@st.cache_resource
def get_driver():
    # --- KONFIGURASI DATABASE NEO4J ---
    uri = "bolt://127.0.0.1:7687" # Default Neo4j Bolt port
    username = "neo4j"
    password = st.secrets["NEO4J_PASS"] # Replace with your Neo4j password
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        # Verify connection
        driver.verify_connectivity()
        return driver
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}. Please check your Neo4j server and credentials.")
        return None

# Initialize driver globally
driver = get_driver()

# Fungsi Embedding, Pencarian, dll. (tidak berubah, saya persingkat untuk fokus pada perubahan avatar)
def get_embedding(question: str, model: str = "text-embedding-3-small"):
    try:
        response = openai.embeddings.create(input=question, model=model)
        return np.array(response.data[0].embedding).astype('float32').reshape(1, -1)
    except Exception as e:
        st.error(f"🧠 **Kesalahan Embedding!** Gagal dari OpenAI: `{e}`.", icon="💡")
        return None

def search_similar_passages(faiss_index, embedding_vector, k: int = 7):
    if faiss_index is None or embedding_vector is None: return None, None
    try: return faiss_index.search(embedding_vector, k)
    except Exception as e:
        st.error(f"🔍 **Kesalahan Pencarian FAISS!** `{e}`", icon="🔦")
        return None, None

def get_top_metadata_ids(indices, metadata_df, column="id_pasal"):
    if indices is None or metadata_df is None: return []
    try: return [metadata_df.iloc[idx].get(column, "Unknown_ID") for idx in indices[0]]
    except IndexError:
        st.warning("⚠️ Indeks FAISS di luar batas metadata.", icon="📊")
        return []
    except Exception as e:
        st.error(f"📄 **Kesalahan Metadata!** `{e}`", icon="📑")
        return []

def get_pasal_details(ids):
    if not ids or driver is None: return []
    ids_str = ",".join([f"'{str(id_val)}'" for id_val in ids])
    query = f"MATCH (doc:Document)-[:CONTAINS]->(bab:Bab)-[:CONTAINS]->(pasal:Pasal) WHERE pasal.id IN [{ids_str}] RETURN doc.id AS doc_id, doc.title AS doc_name, bab.id AS bab_id, bab.name AS bab_name, pasal.id AS pasal_id, pasal.name AS pasal_name, pasal.content AS pasal_content ORDER BY doc.title, bab.name, pasal.name"
    try:
        with driver.session() as session: return [record.data() for record in session.run(query)]
    except Exception as e:
        st.error(f"🔗 **Kesalahan Query Neo4j!** `{e}`", icon="💾")
        return []

def build_prompt(pasal_records, question):
    prompt_parts = ["Anda adalah Asisten Hukum AI...\n"] # Disingkat
    if not pasal_records: prompt_parts.append("Tidak ada pasal relevan...\n")
    else:
        for record in pasal_records:
            prompt_parts.append(f"Dokumen: {record.get('doc_name', 'N/A')}...\nIsi Pasal: {record.get('pasal_content', 'N/A')}\n---\n")
    prompt_parts.append(f"\n**Pertanyaan Pengguna:**\n{question}\n")
    prompt_parts.append("\n**Instruksi untuk Asisten Hukum AI:**\n1. Jawab HANYA berdasarkan informasi...\n") # Disingkat
    return "\n".join(prompt_parts)


# Inisialisasi riwayat chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    current_avatar = USER_AVATAR_SVG if msg["role"] == "user" else AI_AVATAR_SVG
    with st.chat_message(msg["role"], avatar=current_avatar):
        st.markdown(msg["content_display"], unsafe_allow_html=msg.get("unsafe_html", False))


# Antarmuka chat utama
if user_prompt := st.chat_input("Ketik pertanyaan hukum Anda di sini..."):
    st.session_state.messages.append({"role": "user", "content_raw": user_prompt, "content_display": user_prompt, "unsafe_html": False})
    # --- MODIFIKASI: Tampilkan pesan pengguna baru dengan avatar kustom ---
    with st.chat_message("user", avatar=USER_AVATAR_SVG):
        st.markdown(user_prompt)

    # --- MODIFIKASI: Giliran Asisten dengan avatar kustom ---
    with st.chat_message("assistant", avatar=AI_AVATAR_SVG):
        if index is None or metadata is None or driver is None:
            error_html = """
                <div class='response-box'>
                    <h4 style='color: #FF6B6B;'>Kesalahan Sistem Kritis</h4> <p style='color: #FFC1C1;'>Sistem tidak dapat memproses permintaan Anda karena komponen penting (FAISS, metadata, atau Neo4j) gagal dimuat. Mohon periksa konfigurasi dan log kesalahan di atas.</p>
                </div>
            """
            st.markdown(error_html, unsafe_allow_html=True)
            st.session_state.messages.append({
                "role": "assistant", "content_raw": "Error: System components not loaded.",
                "content_display": error_html, "unsafe_html": True
            })
        else:
            spinner_text = "🔍"
            with st.spinner(spinner_text):
                question_vector = get_embedding(user_prompt)
                answer_from_llm = ""
                
                if question_vector is not None:
                    _, faiss_indices = search_similar_passages(index, question_vector)
                    if faiss_indices is not None and len(faiss_indices[0]) > 0 :
                        top_ids = get_top_metadata_ids(faiss_indices, metadata)
                        if top_ids:
                            results_neo4j = get_pasal_details(top_ids)
                            
                            full_prompt_for_llm = build_prompt(results_neo4j, user_prompt)
                            try:
                                response = openai.chat.completions.create(
                                    model="gpt-4.1-nano",
                                    messages=[
                                        {"role": "system", "content": "Anda adalah asisten hukum AI yang sangat teliti..."},
                                        {"role": "user", "content": full_prompt_for_llm}
                                    ]
                                )
                                answer_from_llm = response.choices[0].message.content
                            except Exception as e:
                                st.error(f"🤖 **Kesalahan API OpenAI!** `{e}`", icon="📡")
                                answer_from_llm = f"Maaf, terjadi kesalahan saat mencoba menghasilkan jawaban dari AI: {e}"
                        else:
                            answer_from_llm = "Tidak ada ID metadata yang relevan ditemukan..."
                            st.info("ℹ️ Tidak ada ID metadata yang relevan...", icon="📂")
                    else:
                        answer_from_llm = "Tidak dapat menemukan bagian yang relevan..."
                        st.info("ℹ️ Tidak dapat menemukan pasal yang relevan...", icon="🔎")
                else:
                    answer_from_llm = "Tidak dapat memproses pertanyaan karena gagal membuat embedding."

                answer_for_html = answer_from_llm.replace("\n", "<br>")

                formatted_answer_html = f"""
                    <div class="response-box">
                        <h4>Jawaban Asisten Hukum</h4>
                        <div>{answer_for_html}</div>
                    </div>
                """

            st.markdown(formatted_answer_html, unsafe_allow_html=True)
            st.session_state.messages.append({
                "role": "assistant", "content_raw": answer_from_llm,
                "content_display": formatted_answer_html, "unsafe_html": True
            })

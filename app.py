import os
import io
import csv
import re
import functools
from pathlib import Path

import pandas as pd
from flask import (Flask, render_template, request, redirect,
                   url_for, session, make_response, send_file)
from transformers import AutoTokenizer, DistilBertForSequenceClassification, pipeline
from bertopic import BERTopic

from database import init_db, get_db

# ==========================================
# OFFLINE MODE
# ==========================================
os.environ['HF_HUB_OFFLINE']      = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

app = Flask(__name__)
app.secret_key = "coretax_skripsi_hamid_2024"

# Password admin — ganti sesuai kebutuhan
ADMIN_PASSWORD = "admin123"

# ==========================================
# PATH KONFIGURASI
# ==========================================
MODEL_DIR  = Path(__file__).parent / "model_skripsi" / "imbalanced" / "skenario_3"
MODEL_PATH = MODEL_DIR.as_posix()
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==========================================
# INIT DATABASE
# ==========================================
init_db()

# ==========================================
# LOAD MODEL
# ==========================================
print("Loading model DistilBERT imbalanced/skenario_3...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}

sentiment_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=False
)
print("Model siap!")

# ==========================================
# AUTH HELPERS
# ==========================================
def is_admin():
    return session.get('is_admin', False)

def admin_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not is_admin():
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated

# ==========================================
# FUNGSI PREDIKSI SENTIMEN
# ==========================================
def light_clean(text: str) -> str:
    """Sama persis dengan light_clean() di dbshm.py — HARUS konsisten dengan
    preprocessing saat training, karena model dilatih pakai full_text + ini,
    bukan clean_text (yang sudah di-stem & dibuang stopword)."""
    text = str(text)
    text = re.sub(r'^\[AUG\]\s*', '', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'\bRT\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def predict_sentiment(full_text):
    try:
        text     = light_clean(full_text)[:512]
        result   = sentiment_pipeline(text)[0]
        label_id = int(result['label'].split('_')[-1])
        return label_map[label_id], round(result['score'] * 100, 1)
    except Exception:
        return "Netral", 0.0

# ==========================================
# FUNGSI BERTOPIC
# ==========================================
def run_bertopic(df):
    periodes  = ["pra_rilis", "sosialisasi", "after_rilis"]
    hasil_all = []

    for periode in periodes:
        df_p = df[df['periode'] == periode].copy()
        docs = df_p['clean_text'].astype(str).tolist()

        if len(docs) < 10:
            continue

        topic_model   = BERTopic(language="multilingual", verbose=False)
        topics, _     = topic_model.fit_transform(docs)
        df_p['topic'] = topics
        topic_info    = topic_model.get_topic_info()

        for _, row in topic_info.iterrows():
            if row['Topic'] == -1:
                continue
            words    = topic_model.get_topic(row['Topic'])
            keywords = ", ".join([w[0] for w in words[:5]])
            df_topik = df_p[df_p['topic'] == row['Topic']]
            sent_dom = df_topik['sentimen_prediksi'].value_counts().idxmax() if len(df_topik) > 0 else "-"
            hasil_all.append({
                "periode"         : periode,
                "topic_id"        : int(row['Topic']),
                "jumlah_tweet"    : int(row['Count']),
                "kata_kunci"      : keywords,
                "sentimen_dominan": sent_dom,
            })

    return hasil_all

# ==========================================
# ROUTES — AUTH
# ==========================================
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(request.args.get('next') or url_for('index'))
        error = "Password salah."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.pop('is_admin', None)
    return redirect(url_for('index'))

# ==========================================
# ROUTES — UPLOAD & HOME
# ==========================================
@app.route("/", methods=["GET"])
def index():
    db      = get_db()
    uploads = db.execute("""
        SELECT u.id, u.filename, u.uploaded_at, u.total_tweets,
               SUM(CASE WHEN t.sentimen_prediksi='Negatif' THEN 1 ELSE 0 END) AS neg_count,
               SUM(CASE WHEN t.sentimen_prediksi='Netral'  THEN 1 ELSE 0 END) AS net_count,
               SUM(CASE WHEN t.sentimen_prediksi='Positif' THEN 1 ELSE 0 END) AS pos_count
        FROM uploads u
        LEFT JOIN tweets t ON t.upload_id = u.id
        GROUP BY u.id
        ORDER BY u.uploaded_at DESC
    """).fetchall()
    db.close()
    return render_template("upload.html",
                           uploads=[dict(u) for u in uploads],
                           is_admin=is_admin())

@app.route("/analisis", methods=["POST"])
@admin_required
def analisis():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filepath = os.path.join(UPLOAD_DIR, "dataset.csv")
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)

        required_cols = {'clean_text', 'periode'}
        missing = required_cols - set(df.columns)
        if missing:
            return _upload_error(f"Kolom tidak ditemukan: {', '.join(missing)}")

        # Filter hanya kolom yang dibutuhkan (ignore kolom lain seperti input_ids, attention_mask, dll)
        if 'full_text' not in df.columns:
            df['full_text'] = df['clean_text']
        
        # Keep hanya kolom penting
        essential_cols = ['full_text', 'clean_text', 'periode', 'created_at']
        for col in essential_cols:
            if col not in df.columns:
                if col == 'created_at':
                    df[col] = ''
                elif col == 'full_text':
                    df[col] = df['clean_text']
        
        df = df[essential_cols]

        # Prediksi sentimen — pakai full_text (light_clean diterapkan di dalam predict_sentiment),
        # konsisten dengan teks yang dipakai saat training model (lihat dbshm.py)
        sentimen_list           = df['full_text'].apply(predict_sentiment)
        df['sentimen_prediksi'] = [s[0] for s in sentimen_list]
        df['confidence']        = [s[1] for s in sentimen_list]

        # Simpan ke DB
        db  = get_db()
        cur = db.execute(
            "INSERT INTO uploads (filename, total_tweets) VALUES (?, ?)",
            (file.filename, len(df))
        )
        upload_id = cur.lastrowid

        for _, row in df.iterrows():
            db.execute(
                """INSERT INTO tweets
                   (upload_id, full_text, clean_text, periode, created_at_tweet,
                    sentimen_prediksi, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (upload_id,
                 str(row.get('full_text', '')),
                 str(row.get('clean_text', '')),
                 str(row.get('periode', '')),
                 str(row.get('created_at', '')),
                 row['sentimen_prediksi'],
                 float(row['confidence']))
            )

        # BERTopic
        hasil_bertopic = run_bertopic(df)
        for r in hasil_bertopic:
            db.execute(
                """INSERT INTO bertopic_results
                   (upload_id, periode, topic_id, jumlah_tweet, kata_kunci, sentimen_dominan)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (upload_id, r['periode'], r['topic_id'],
                 r['jumlah_tweet'], r['kata_kunci'], r['sentimen_dominan'])
            )

        db.commit()
        db.close()

        return redirect(url_for('sentimen_page'))

    except Exception as e:
        return _upload_error(f"Error saat analisis: {str(e)}")


def _upload_error(msg):
    db      = get_db()
    uploads = db.execute("""
        SELECT u.id, u.filename, u.uploaded_at, u.total_tweets,
               SUM(CASE WHEN t.sentimen_prediksi='Negatif' THEN 1 ELSE 0 END) AS neg_count,
               SUM(CASE WHEN t.sentimen_prediksi='Netral'  THEN 1 ELSE 0 END) AS net_count,
               SUM(CASE WHEN t.sentimen_prediksi='Positif' THEN 1 ELSE 0 END) AS pos_count
        FROM uploads u LEFT JOIN tweets t ON t.upload_id = u.id
        GROUP BY u.id ORDER BY u.uploaded_at DESC
    """).fetchall()
    db.close()
    return render_template("upload.html", error=msg,
                           uploads=[dict(u) for u in uploads], is_admin=True)

# ==========================================
# ROUTES — SENTIMEN (akumulasi, pagination, filter)
# ==========================================
@app.route("/sentimen")
def sentimen_page():
    db = get_db()

    total_all = db.execute("SELECT COUNT(*) FROM tweets").fetchone()[0]
    if total_all == 0:
        db.close()
        return redirect(url_for('index'))

    # Global stats (selalu dari semua data)
    dist_rows     = db.execute(
        "SELECT sentimen_prediksi, COUNT(*) as cnt FROM tweets GROUP BY sentimen_prediksi"
    ).fetchall()
    dist_sentimen = {r['sentimen_prediksi']: r['cnt'] for r in dist_rows}

    dist_periode = {}
    for p in ['pra_rilis', 'sosialisasi', 'after_rilis']:
        rows = db.execute(
            "SELECT sentimen_prediksi, COUNT(*) as cnt FROM tweets WHERE periode=? GROUP BY sentimen_prediksi",
            (p,)
        ).fetchall()
        dist_periode[p] = {r['sentimen_prediksi']: r['cnt'] for r in rows}

    dominant_sentimen = max(dist_sentimen, key=dist_sentimen.get) if dist_sentimen else '-'
    dom_row = db.execute(
        "SELECT periode FROM tweets WHERE sentimen_prediksi=? GROUP BY periode ORDER BY COUNT(*) DESC LIMIT 1",
        (dominant_sentimen,)
    ).fetchone()
    dominant_periode = dom_row['periode'] if dom_row else '-'

    # Pagination + filter (sentimen & periode)
    page      = max(1, request.args.get('page', 1, type=int))
    per_page  = 25
    filter_s  = request.args.get('filter', 'all')
    filter_p  = request.args.get('periode', 'all')

    VALID_SENTS   = {'Negatif', 'Netral', 'Positif'}
    VALID_PERIODES = {'pra_rilis', 'sosialisasi', 'after_rilis'}
    if filter_s  not in VALID_SENTS:    filter_s  = 'all'
    if filter_p  not in VALID_PERIODES: filter_p  = 'all'

    # Build WHERE conditions
    conds  = []
    params = []
    if filter_s != 'all':
        conds.append("sentimen_prediksi = ?")
        params.append(filter_s)
    if filter_p != 'all':
        conds.append("periode = ?")
        params.append(filter_p)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""

    total_filtered = db.execute(
        f"SELECT COUNT(*) FROM tweets {where}", params
    ).fetchone()[0]

    tweet_rows = db.execute(
        f"SELECT * FROM tweets {where} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [per_page, (page - 1) * per_page]
    ).fetchall()

    db.close()

    total_pages = max(1, (total_filtered + per_page - 1) // per_page)
    page        = min(page, total_pages)

    page_start = max(1, page - 2)
    page_end   = min(total_pages, page + 2)
    page_range = list(range(page_start, page_end + 1))

    return render_template(
        "sentimen.html",
        total             = total_all,
        dist_sentimen     = dist_sentimen,
        dist_periode      = dist_periode,
        dominant_sentimen = dominant_sentimen,
        dominant_periode  = dominant_periode,
        tweets_page       = [dict(t) for t in tweet_rows],
        page              = page,
        total_pages       = total_pages,
        page_range        = page_range,
        filter_s          = filter_s,
        filter_p          = filter_p,
        total_filtered    = total_filtered,
    )

# ==========================================
# ROUTES — BERTOPIC (dari upload terakhir)
# ==========================================
@app.route("/bertopic")
def bertopic_page():
    db = get_db()

    latest = db.execute(
        "SELECT id FROM uploads ORDER BY uploaded_at DESC LIMIT 1"
    ).fetchone()
    if not latest:
        db.close()
        return redirect(url_for('index'))

    upload_id = latest['id']
    periodes  = ["pra_rilis", "sosialisasi", "after_rilis"]
    data      = {}
    summary   = {}

    for p in periodes:
        rows    = db.execute(
            "SELECT * FROM bertopic_results WHERE upload_id=? AND periode=? ORDER BY jumlah_tweet DESC",
            (upload_id, p)
        ).fetchall()
        data[p] = [dict(r) for r in rows]
        if rows:
            sc = {}
            for r in rows:
                s = r['sentimen_dominan']
                sc[s] = sc.get(s, 0) + 1
            summary[p] = {
                'jumlah_topik'    : len(rows),
                'sentimen_dominan': max(sc, key=sc.get),
            }
        else:
            summary[p] = {'jumlah_topik': 0, 'sentimen_dominan': '-'}

    top_by_sentimen = {}
    for sent in ['Negatif', 'Netral', 'Positif']:
        row = db.execute(
            "SELECT * FROM bertopic_results WHERE upload_id=? AND sentimen_dominan=? ORDER BY jumlah_tweet DESC LIMIT 1",
            (upload_id, sent)
        ).fetchone()
        if row:
            top_by_sentimen[sent] = dict(row)

    db.close()
    return render_template("bertopic.html", data=data, summary=summary, top_by_sentimen=top_by_sentimen)

# ==========================================
# ROUTES — DOWNLOAD
# ==========================================
@app.route("/download/sentimen.csv")
def download_sentimen():
    db   = get_db()
    rows = db.execute(
        "SELECT full_text, clean_text, periode, created_at_tweet, sentimen_prediksi, confidence FROM tweets ORDER BY id"
    ).fetchall()
    db.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['full_text', 'clean_text', 'periode', 'created_at', 'sentimen_prediksi', 'confidence'])
    for r in rows:
        writer.writerow(list(r))

    resp = make_response(output.getvalue())
    resp.headers["Content-Disposition"] = "attachment; filename=hasil_sentimen.csv"
    resp.headers["Content-Type"]        = "text/csv; charset=utf-8"
    return resp

@app.route("/download/db")
@admin_required
def download_db():
    from database import DB_PATH
    return send_file(DB_PATH, as_attachment=True, download_name="coretax.db")

if __name__ == "__main__":
    app.run(debug=False)


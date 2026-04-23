"""
Campbell's Restaurant — AI Marketing System
FastAPI Backend with Supabase Integration

Endpoints:
  GET  /                        → health check
  POST /api/segment             → customer segment + RFM
  POST /api/churn               → churn probability + risk level
  POST /api/sentiment           → ABSA triplets from review text
  POST /api/generate-message    → personalized SMS + email + app notification
  GET  /api/dashboard           → full stats for frontend dashboard (live from Supabase)
  POST /api/full-pipeline       → all of the above in one call
  GET  /api/customers           → paginated customer list from Supabase
  GET  /api/messages-log        → recent generated messages log
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import pickle
import json
import re
import os
import requests
import warnings
from datetime import datetime, timezone
from supabase import create_client, Client
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier
from openpyxl import load_workbook
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title       = "Campbell's AI Marketing API",
    description = "Churn Prediction · Customer Segmentation · ABSA · Personalized Messages",
    version     = "2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # restrict in production to your Lovable domain
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY",    "YOUR_GROQ_API_KEY_HERE")
SUPABASE_URL    = os.getenv("SUPABASE_URL",    "YOUR_SUPABASE_URL_HERE")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY",    "YOUR_SUPABASE_ANON_KEY_HERE")
GROQ_MODEL      = "llama-3.3-70b-versatile"
DATA_DIR        = os.getenv("DATA_DIR", ".")
DISCOUNT_MAP    = {'High': 20, 'Medium': 15, 'Low': 10}

# ─────────────────────────────────────────────
# SUPABASE CLIENT
# ─────────────────────────────────────────────
supabase: Client = None

def get_supabase() -> Client:
    global supabase
    if supabase is None:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase

# ─────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────
class CustomerRequest(BaseModel):
    recency         : float
    frequency       : int
    monetary        : float
    unique_items    : Optional[int]   = 1
    avg_order_val   : Optional[float] = None
    avg_tip         : Optional[float] = 0.0
    discount_used   : Optional[int]   = 0
    visits_nov      : Optional[int]   = 0
    visits_dec      : Optional[int]   = 0
    visits_jan      : Optional[int]   = 0
    days_since_first: Optional[int]   = None
    favorite_items  : Optional[List[str]] = []

class SentimentRequest(BaseModel):
    review: str

class MessageRequest(BaseModel):
    customer_id      : Optional[str]  = None
    segment          : str
    recency          : float
    frequency        : int
    monetary         : float
    risk_level       : str
    churn_probability: float
    favorite_items   : Optional[List[str]] = []
    aspects          : Optional[List[str]] = []
    sentiments       : Optional[List[str]] = []

class FullPipelineRequest(BaseModel):
    recency         : float
    frequency       : int
    monetary        : float
    unique_items    : Optional[int]   = 1
    avg_order_val   : Optional[float] = None
    avg_tip         : Optional[float] = 0.0
    discount_used   : Optional[int]   = 0
    visits_nov      : Optional[int]   = 0
    visits_dec      : Optional[int]   = 0
    visits_jan      : Optional[int]   = 0
    days_since_first: Optional[int]   = None
    favorite_items  : Optional[List[str]] = []
    review          : Optional[str]   = None
    customer_id     : Optional[str]   = None

# ─────────────────────────────────────────────
# MODEL STORE
# ─────────────────────────────────────────────
models = {}

def load_models():
    global models
    try:
        with open(f"{DATA_DIR}/kmeans_model.pkl",      'rb') as f: models['kmeans']      = pickle.load(f)
        with open(f"{DATA_DIR}/scaler.pkl",            'rb') as f: models['scaler']      = pickle.load(f)
        with open(f"{DATA_DIR}/cluster_map.pkl",       'rb') as f: models['cluster_map'] = pickle.load(f)
        with open(f"{DATA_DIR}/churn_model_tier2.pkl", 'rb') as f: models['churn']       = pickle.load(f)
        with open(f"{DATA_DIR}/churn_features.pkl",    'rb') as f: models['churn_feats'] = pickle.load(f)
        with open(f"{DATA_DIR}/aspect_model.pkl",      'rb') as f: models['aspect']      = pickle.load(f)
        with open(f"{DATA_DIR}/tfidf_aspect.pkl",      'rb') as f: models['tfidf_asp']   = pickle.load(f)
        with open(f"{DATA_DIR}/mlb.pkl",               'rb') as f: models['mlb']         = pickle.load(f)
        with open(f"{DATA_DIR}/sentiment_model.pkl",   'rb') as f: models['sentiment']   = pickle.load(f)
        with open(f"{DATA_DIR}/tfidf_sent.pkl",        'rb') as f: models['tfidf_sent']  = pickle.load(f)
        print("✅ All models loaded successfully")
    except FileNotFoundError as e:
        print(f"⚠️  Model file not found: {e}. Run the notebook first to generate pickle files.")

def load_menu():
    try:
        wb      = load_workbook(f"{DATA_DIR}/Campbell_Menu_Data_-_2.xlsx", read_only=True)
        ws      = wb.active
        rows    = list(ws.iter_rows(values_only=True))
        menu_df = pd.DataFrame(rows[1:], columns=rows[0])
        food_cats = ['Signature Flights','Brunch Food','Entrées','Desserts',
                     'Salads','Burgers & Sandwiches','Kids Menu',
                     'Sides Dinner','Sides Brunch','Weekly Specials']
        return menu_df[menu_df['Category'].isin(food_cats)].dropna(subset=['itemName','itemPrice'])
    except:
        return pd.DataFrame(columns=['itemName','itemPrice','Category'])

menu_df = pd.DataFrame()

# ─────────────────────────────────────────────
# SUPABASE — TABLE CREATION
# ─────────────────────────────────────────────

CREATE_CUSTOMERS_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    id                  TEXT PRIMARY KEY,
    segment             TEXT,
    recency             FLOAT,
    frequency           INTEGER,
    monetary            FLOAT,
    unique_items        INTEGER,
    avg_order_val       FLOAT,
    avg_tip             FLOAT,
    discount_used       INTEGER,
    visits_nov          INTEGER,
    visits_dec          INTEGER,
    visits_jan          INTEGER,
    days_since_first    INTEGER,
    churn_probability   FLOAT,
    risk_level          TEXT,
    tier                TEXT,
    discount_offered    TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_MESSAGES_LOG_SQL = """
CREATE TABLE IF NOT EXISTS messages_log (
    id               BIGSERIAL PRIMARY KEY,
    customer_id      TEXT REFERENCES customers(id) ON DELETE SET NULL,
    segment          TEXT,
    risk_level       TEXT,
    discount_offered TEXT,
    sms              TEXT,
    email_subject    TEXT,
    email_body       TEXT,
    app_notification TEXT,
    aspects          JSONB,
    sentiments       JSONB,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_ABSA_SQL = """
CREATE TABLE IF NOT EXISTS absa_predictions (
    id          BIGSERIAL PRIMARY KEY,
    customer_id TEXT,
    review      TEXT,
    aspects     JSONB,
    sentiments  JSONB,
    opinions    JSONB,
    triplets    JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
"""

def create_tables():
    """Create Supabase tables via raw SQL using the REST API."""
    sb = get_supabase()
    sql_statements = [
        CREATE_CUSTOMERS_SQL,
        CREATE_MESSAGES_LOG_SQL,
        CREATE_ABSA_SQL,
    ]
    for sql in sql_statements:
        try:
            sb.rpc("exec_sql", {"query": sql}).execute()
        except Exception as e:
            # Tables might already exist — try direct approach via postgrest
            print(f"ℹ️  Table creation note: {e}")
    print("✅ Supabase tables ready")

def ensure_tables_via_api():
    """
    Alternative: Create tables using Supabase Management API.
    Uses SUPABASE_SERVICE_KEY (not anon key) for admin operations.

    If you have your service_role key set as SUPABASE_SERVICE_KEY,
    this runs raw SQL directly. Otherwise tables must be created once
    via the Supabase dashboard SQL editor using the SQL above.
    """
    service_key = os.getenv("SUPABASE_SERVICE_KEY", "")
    project_ref = SUPABASE_URL.replace("https://", "").replace(".supabase.co", "")

    if not service_key:
        print("⚠️  SUPABASE_SERVICE_KEY not set — skipping auto table creation.")
        print("    Run this SQL once in your Supabase SQL editor:")
        print("    " + CREATE_CUSTOMERS_SQL[:80] + "...")
        return False

    all_sql = "\n".join([CREATE_CUSTOMERS_SQL, CREATE_MESSAGES_LOG_SQL, CREATE_ABSA_SQL])
    url     = f"https://api.supabase.com/v1/projects/{project_ref}/database/query"
    headers = {
        "Authorization": f"Bearer {service_key}",
        "Content-Type" : "application/json"
    }
    try:
        resp = requests.post(url, headers=headers, json={"query": all_sql}, timeout=30)
        if resp.status_code in (200, 201):
            print("✅ Tables created via Management API")
            return True
        else:
            print(f"⚠️  Management API returned {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"⚠️  Management API error: {e}")
        return False

# ─────────────────────────────────────────────
# SUPABASE — SEED CUSTOMER DATA FROM CSV
# ─────────────────────────────────────────────

def seed_customers_from_csv():
    """
    Load churn_scores_final.csv and upsert all rows into Supabase customers table.
    Safe to call multiple times — uses upsert so no duplicates.
    """
    csv_path = f"{DATA_DIR}/churn_scores_final.csv"
    if not os.path.exists(csv_path):
        print(f"⚠️  {csv_path} not found — skipping customer seed. Run notebook first.")
        return 0

    df = pd.read_csv(csv_path)
    print(f"📦 Seeding {len(df)} customers to Supabase...")

    # Normalise column names — adapt to whatever your CSV actually has
    col_map = {
        'Last 4 Card Digits': 'id',
        'Segment'            : 'segment',
        'Recency'            : 'recency',
        'Frequency'          : 'frequency',
        'Monetary'           : 'monetary',
        'Unique_Items'       : 'unique_items',
        'Avg_Order_Val'      : 'avg_order_val',
        'Avg_Tip'            : 'avg_tip',
        'Discount_Used'      : 'discount_used',
        'Visits_Nov'         : 'visits_nov',
        'Visits_Dec'         : 'visits_dec',
        'Visits_Jan'         : 'visits_jan',
        'Days_Since_First'   : 'days_since_first',
        'Churn_Probability'  : 'churn_probability',
        'Risk_Level'         : 'risk_level',
        'Tier'               : 'tier',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure required columns exist with defaults
    required = {
        'id': lambda i: f"customer_{i}",
        'segment': 'Unknown', 'recency': 0.0, 'frequency': 0,
        'monetary': 0.0, 'unique_items': 1, 'avg_order_val': 0.0,
        'avg_tip': 0.0, 'discount_used': 0, 'visits_nov': 0,
        'visits_dec': 0, 'visits_jan': 0, 'days_since_first': 0,
        'churn_probability': 0.0, 'risk_level': 'Low', 'tier': 'Unknown',
    }
    for col, default in required.items():
        if col not in df.columns:
            df[col] = [default(i) if callable(default) else default for i in range(len(df))]

    df['id']               = df['id'].astype(str).str.strip()
    df['discount_offered'] = df['risk_level'].map(lambda r: f"{DISCOUNT_MAP.get(str(r), 15)}%")

    # Fill NaNs to avoid JSON serialization issues
    df = df.fillna(0)

    keep_cols = list(required.keys()) + ['discount_offered']
    df        = df[[c for c in keep_cols if c in df.columns]]

    # Batch upsert in chunks of 500
    records = df.to_dict('records')
    sb      = get_supabase()
    batch   = 500
    seeded  = 0
    for i in range(0, len(records), batch):
        chunk = records[i:i+batch]
        # Convert all numpy types to native Python
        chunk = [
            {k: (int(v) if isinstance(v, (np.integer,)) else
                 float(v) if isinstance(v, (np.floating,)) else v)
             for k, v in row.items()}
            for row in chunk
        ]
        try:
            sb.table("customers").upsert(chunk, on_conflict="id").execute()
            seeded += len(chunk)
        except Exception as e:
            print(f"⚠️  Batch {i//batch + 1} upsert error: {e}")

    print(f"✅ Seeded {seeded} customers into Supabase")
    return seeded

# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global menu_df
    load_models()
    menu_df = load_menu()
    print(f"✅ Menu loaded: {len(menu_df)} items")

    # Create tables then seed data
    tables_ok = ensure_tables_via_api()
    if not tables_ok:
        print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTION REQUIRED — Run this SQL once in your Supabase SQL Editor:
https://supabase.com/dashboard/project/_/sql

""" + CREATE_CUSTOMERS_SQL + "\n" + CREATE_MESSAGES_LOG_SQL + "\n" + CREATE_ABSA_SQL + """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)

    # Always try to seed — upsert is idempotent
    seed_customers_from_csv()

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_segment(recency, frequency, monetary) -> str:
    if 'kmeans' not in models:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")
    scaled  = models['scaler'].transform([[recency, frequency, monetary]])
    cluster = models['kmeans'].predict(scaled)[0]
    return models['cluster_map'].get(cluster, 'Unknown')

def get_churn_score(req: CustomerRequest, segment: str) -> dict:
    if req.frequency == 1:
        if req.recency < 14:   prob = 0.30
        elif req.recency < 30: prob = 0.65
        else:                  prob = 0.92
        tier = "Rule-Based"
    else:
        if 'churn' not in models:
            raise HTTPException(status_code=503, detail="Churn model not loaded")
        seg_map  = {'Regular': 0, 'New': 1, 'Occasional': 2, 'Lost': 3}
        features = {
            'Recency'        : req.recency,
            'Frequency'      : req.frequency,
            'Monetary'       : req.monetary,
            'Unique_Items'   : req.unique_items or 1,
            'Avg_Order_Val'  : req.avg_order_val or req.monetary,
            'Avg_Tip'        : req.avg_tip or 0.0,
            'Discount_Used'  : req.discount_used or 0,
            'Visits_Nov'     : req.visits_nov or 0,
            'Visits_Dec'     : req.visits_dec or 0,
            'Visits_Jan'     : req.visits_jan or 0,
            'Days_Since_First': req.days_since_first or int(req.recency),
            'Segment_Code'   : seg_map.get(segment, 1)
        }
        X    = pd.DataFrame([features])[models['churn_feats']]
        prob = float(models['churn'].predict_proba(X)[0][1])
        tier = "XGBoost"

    if prob < 0.33:   risk = "Low"
    elif prob < 0.66: risk = "Medium"
    else:             risk = "High"

    return {"churn_probability": round(prob, 4), "risk_level": risk, "tier": tier}

OPINION_KEYWORDS = {
    'food'    : ['food','dish','meal','taste','flavor','delicious','bland','overcooked','fresh','portion'],
    'staff'   : ['waiter','waitress','server','staff','host','bartender','friendly','rude','helpful','attentive'],
    'service' : ['service','wait','slow','fast','quick','prompt','attentive','responsive'],
    'place'   : ['place','location','restaurant','spot','venue','seating','atmosphere'],
    'menu'    : ['menu','options','variety','selection','choice','specials'],
    'ambience': ['ambience','ambiance','atmosphere','decor','noise','cozy','loud','vibe','setting'],
    'price'   : ['price','expensive','cheap','value','worth','overpriced','affordable','cost']
}

def extract_opinion(review: str, aspect: str) -> str:
    keywords  = OPINION_KEYWORDS.get(aspect, [aspect])
    sentences = re.split(r'[.!?,;]', review)
    for sent in sentences:
        if any(kw in sent.lower() for kw in keywords):
            words = sent.strip().split()
            for i, word in enumerate(words):
                if word.lower().strip('.,!?') in keywords:
                    snippet = ' '.join(words[max(0,i-3):min(len(words),i+4)]).strip('.,!? ')
                    if snippet: return snippet
    return aspect

def run_absa(review: str) -> dict:
    if 'aspect' not in models:
        raise HTTPException(status_code=503, detail="ABSA model not loaded")
    X_r       = models['tfidf_asp'].transform([review])
    aspects   = list(models['mlb'].inverse_transform(models['aspect'].predict(X_r))[0])
    sentiments, opinions, triplets = [], [], []
    for asp in aspects:
        inp       = models['tfidf_sent'].transform([review + ' [ASPECT] ' + asp])
        sentiment = models['sentiment'].predict(inp)[0]
        opinion   = extract_opinion(review, asp)
        sentiments.append(sentiment)
        opinions.append(opinion)
        triplets.append({"aspect": asp, "opinion": opinion, "sentiment": sentiment})
    return {"aspects": aspects, "sentiments": sentiments, "opinions": opinions, "triplets": triplets}

def call_groq(prompt: str) -> str:
    headers = {
        "Content-Type" : "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    body = {
        "model"          : GROQ_MODEL,
        "max_tokens"     : 1000,
        "temperature"    : 0.7,
        "response_format": {"type": "json_object"},
        "messages"       : [
            {"role": "system", "content": "You are a marketing assistant for Campbell's Restaurant. Respond with valid JSON only."},
            {"role": "user",   "content": prompt}
        ]
    }
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers, json=body, timeout=30
    )
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']

def generate_message(req: MessageRequest) -> dict:
    discount   = DISCOUNT_MAP.get(req.risk_level, 15)
    liked      = [a for a, s in zip(req.aspects or [], req.sentiments or []) if s == 'positive']
    disliked   = [a for a, s in zip(req.aspects or [], req.sentiments or []) if s == 'negative']
    highlights = []
    if not menu_df.empty:
        highlights = menu_df.sample(min(3, len(menu_df)))[['itemName','itemPrice','Category']].to_dict('records')

    prompt = f"""You are a warm, friendly marketing assistant for Campbell's Restaurant.

CUSTOMER PROFILE:
- Segment            : {req.segment}
- Days since last visit: {int(req.recency)} days
- Total visits       : {req.frequency}
- Total spent        : ${req.monetary:.2f}
- Churn risk         : {req.risk_level}
- Liked aspects      : {liked if liked else 'unknown'}
- Disliked aspects   : {disliked if disliked else 'none noted'}
- Favorite items     : {req.favorite_items}

MENU HIGHLIGHTS:
{json.dumps(highlights, indent=2)}

DISCOUNT TO OFFER: {discount}% off next visit

Return ONLY this JSON:
{{
  "sms": "...",
  "email": {{"subject": "...", "body": "..."}},
  "app_notification": "..."
}}

RULES:
- SMS: max 160 chars, casual and punchy
- Email: warm, personal, 3-4 short paragraphs
- App notification: max 80 chars, exciting
- Tone: Lost→urgent ("we miss you!"), New→welcoming, Occasional→appreciative, Regular→VIP
- NEVER mention churn, AI, or risk scores"""

    response = call_groq(prompt)
    try:
        clean      = re.sub(r'```json|```', '', response).strip()
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if json_match: clean = json_match.group(0)
        return json.loads(clean)
    except:
        return {"sms": response[:160], "email": {}, "app_notification": response[:80]}

# ─────────────────────────────────────────────
# SUPABASE — WRITE HELPERS
# ─────────────────────────────────────────────

def upsert_customer_to_db(customer_id: str, segment: str, req, churn: dict):
    """Upsert a single customer record (called from full-pipeline)."""
    if not customer_id:
        return
    sb = get_supabase()
    record = {
        "id"               : str(customer_id),
        "segment"          : segment,
        "recency"          : float(req.recency),
        "frequency"        : int(req.frequency),
        "monetary"         : float(req.monetary),
        "unique_items"     : int(req.unique_items or 1),
        "avg_order_val"    : float(req.avg_order_val or req.monetary),
        "avg_tip"          : float(req.avg_tip or 0),
        "discount_used"    : int(req.discount_used or 0),
        "visits_nov"       : int(req.visits_nov or 0),
        "visits_dec"       : int(req.visits_dec or 0),
        "visits_jan"       : int(req.visits_jan or 0),
        "days_since_first" : int(req.days_since_first or req.recency),
        "churn_probability": float(churn['churn_probability']),
        "risk_level"       : churn['risk_level'],
        "tier"             : churn['tier'],
        "discount_offered" : f"{DISCOUNT_MAP.get(churn['risk_level'], 15)}%",
        "updated_at"       : datetime.now(timezone.utc).isoformat(),
    }
    try:
        sb.table("customers").upsert(record, on_conflict="id").execute()
    except Exception as e:
        print(f"⚠️  Customer upsert failed: {e}")

def log_message_to_db(customer_id: Optional[str], req: MessageRequest, messages: dict, absa: Optional[dict]):
    """Insert a generated message into messages_log."""
    sb = get_supabase()
    email = messages.get("email", {})
    record = {
        "customer_id"     : str(customer_id) if customer_id else None,
        "segment"         : req.segment,
        "risk_level"      : req.risk_level,
        "discount_offered": f"{DISCOUNT_MAP.get(req.risk_level, 15)}%",
        "sms"             : messages.get("sms", ""),
        "email_subject"   : email.get("subject", "") if isinstance(email, dict) else "",
        "email_body"      : email.get("body", "")    if isinstance(email, dict) else "",
        "app_notification": messages.get("app_notification", ""),
        "aspects"         : json.dumps(absa['aspects']    if absa else []),
        "sentiments"      : json.dumps(absa['sentiments'] if absa else []),
    }
    try:
        sb.table("messages_log").insert(record).execute()
    except Exception as e:
        print(f"⚠️  Message log insert failed: {e}")

def log_absa_to_db(customer_id: Optional[str], review: str, absa: dict):
    """Insert ABSA result into absa_predictions."""
    sb = get_supabase()
    record = {
        "customer_id": str(customer_id) if customer_id else None,
        "review"     : review,
        "aspects"    : json.dumps(absa['aspects']),
        "sentiments" : json.dumps(absa['sentiments']),
        "opinions"   : json.dumps(absa['opinions']),
        "triplets"   : json.dumps(absa['triplets']),
    }
    try:
        sb.table("absa_predictions").insert(record).execute()
    except Exception as e:
        print(f"⚠️  ABSA log insert failed: {e}")

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status"       : "ok",
        "service"      : "Campbell's AI Marketing API",
        "version"      : "2.0.0",
        "models_loaded": list(models.keys()),
        "supabase"     : SUPABASE_URL != "YOUR_SUPABASE_URL_HERE"
    }

# ── Segmentation ──
@app.post("/api/segment")
def segment_customer(req: CustomerRequest):
    segment = get_segment(req.recency, req.frequency, req.monetary)
    return {"segment": segment, "recency": req.recency, "frequency": req.frequency, "monetary": req.monetary}

# ── Churn Prediction ──
@app.post("/api/churn")
def predict_churn(req: CustomerRequest):
    segment = get_segment(req.recency, req.frequency, req.monetary)
    result  = get_churn_score(req, segment)
    return {
        "segment"          : segment,
        "churn_probability": result['churn_probability'],
        "risk_level"       : result['risk_level'],
        "tier"             : result['tier'],
        "discount_to_offer": f"{DISCOUNT_MAP.get(result['risk_level'], 15)}%"
    }

# ── Sentiment Analysis ──
@app.post("/api/sentiment")
def analyze_sentiment(req: SentimentRequest):
    result = run_absa(req.review)
    return {
        "review"    : req.review,
        "aspects"   : result['aspects'],
        "sentiments": result['sentiments'],
        "opinions"  : result['opinions'],
        "triplets"  : result['triplets']
    }

# ── Message Generation ──
@app.post("/api/generate-message")
def generate_personalized_message(req: MessageRequest):
    messages = generate_message(req)
    # Log to Supabase
    log_message_to_db(req.customer_id, req, messages, None)
    return {
        "customer_id"     : req.customer_id,
        "segment"         : req.segment,
        "risk_level"      : req.risk_level,
        "discount_offered": f"{DISCOUNT_MAP.get(req.risk_level, 15)}%",
        "messages"        : messages
    }

# ── Full Pipeline ──
@app.post("/api/full-pipeline")
def full_pipeline(req: FullPipelineRequest):
    # Step 1 — Segment
    segment = get_segment(req.recency, req.frequency, req.monetary)

    # Step 2 — Churn
    churn_req = CustomerRequest(**req.dict(exclude={"review", "customer_id"}))
    churn     = get_churn_score(churn_req, segment)

    # Step 3 — ABSA (optional)
    absa = None
    if req.review:
        absa = run_absa(req.review)
        log_absa_to_db(req.customer_id, req.review, absa)

    # Step 4 — Generate message
    msg_req = MessageRequest(
        customer_id      = req.customer_id,
        segment          = segment,
        recency          = req.recency,
        frequency        = req.frequency,
        monetary         = req.monetary,
        risk_level       = churn['risk_level'],
        churn_probability= churn['churn_probability'],
        favorite_items   = req.favorite_items,
        aspects          = absa['aspects']    if absa else [],
        sentiments       = absa['sentiments'] if absa else []
    )
    messages = generate_message(msg_req)

    # Persist to Supabase
    upsert_customer_to_db(req.customer_id, segment, req, churn)
    log_message_to_db(req.customer_id, msg_req, messages, absa)

    return {
        "segment"          : segment,
        "churn_probability": churn['churn_probability'],
        "risk_level"       : churn['risk_level'],
        "tier"             : churn['tier'],
        "discount_offered" : f"{DISCOUNT_MAP.get(churn['risk_level'], 15)}%",
        "absa"             : absa,
        "messages"         : messages
    }

# ── Dashboard Stats (live from Supabase) ──
@app.get("/api/dashboard")
def get_dashboard_stats():
    """
    Return aggregated stats for the Lovable dashboard.
    Pulls live data from Supabase customers table.
    Falls back to CSV if Supabase is not configured.
    """
    sb = get_supabase()

    try:
        # Fetch all customers (up to 10k rows)
        resp = sb.table("customers").select("segment, risk_level, recency, frequency, monetary, churn_probability").limit(10000).execute()
        rows = resp.data

        if not rows:
            raise ValueError("No data in Supabase yet")

        df = pd.DataFrame(rows)

    except Exception as e:
        print(f"⚠️  Supabase dashboard query failed: {e} — falling back to CSV")
        try:
            df = pd.read_csv(f"{DATA_DIR}/churn_scores_final.csv")
            col_map = {
                'Segment': 'segment', 'Risk_Level': 'risk_level',
                'Recency': 'recency', 'Frequency': 'frequency',
                'Monetary': 'monetary', 'Churn_Probability': 'churn_probability'
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="No data source available. Configure Supabase or run notebook.")

    seg_counts  = df['segment'].value_counts().to_dict()   if 'segment'   in df.columns else {}
    risk_counts = df['risk_level'].value_counts().to_dict() if 'risk_level' in df.columns else {}

    avg_churn = float(df['churn_probability'].mean()) if 'churn_probability' in df.columns else 0
    churn_rate = round(avg_churn * 100, 2)

    top_risk = (
        df[df['risk_level'].astype(str) == 'High']
        .sort_values('churn_probability', ascending=False)
        .head(20)
        [['segment','recency','frequency','monetary','churn_probability','risk_level']]
        .to_dict('records')
    ) if 'risk_level' in df.columns else []

    rfm_by_segment = {}
    if 'segment' in df.columns:
        rfm_by_segment = (
            df.groupby('segment')[['recency','frequency','monetary']]
            .mean().round(2)
            .to_dict('index')
        )

    # Recent messages count
    recent_messages = 0
    try:
        msg_resp        = sb.table("messages_log").select("id", count="exact").execute()
        recent_messages = msg_resp.count or 0
    except:
        pass

    return {
        "total_customers"  : int(len(df)),
        "segment_counts"   : seg_counts,
        "risk_counts"      : risk_counts,
        "churn_rate_pct"   : churn_rate,
        "top_at_risk"      : top_risk,
        "rfm_by_segment"   : rfm_by_segment,
        "messages_sent"    : recent_messages,
        "data_source"      : "supabase"
    }

# ── Customers List (paginated) ──
@app.get("/api/customers")
def list_customers(
    segment   : Optional[str] = Query(None, description="Filter by segment"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level: Low/Medium/High"),
    page      : int           = Query(1,    ge=1, description="Page number"),
    page_size : int           = Query(50,   ge=1, le=200, description="Rows per page")
):
    """
    Return paginated customer list from Supabase.
    Supports filtering by segment and/or risk_level.
    """
    sb     = get_supabase()
    offset = (page - 1) * page_size

    query = sb.table("customers").select("*", count="exact")
    if segment:
        query = query.eq("segment", segment)
    if risk_level:
        query = query.eq("risk_level", risk_level)

    try:
        resp = query.range(offset, offset + page_size - 1).execute()
        return {
            "page"         : page,
            "page_size"    : page_size,
            "total"        : resp.count,
            "total_pages"  : -(-resp.count // page_size) if resp.count else 0,
            "customers"    : resp.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase query failed: {e}")

# ── Messages Log ──
@app.get("/api/messages-log")
def get_messages_log(
    limit : int           = Query(50, ge=1, le=200),
    segment: Optional[str] = Query(None)
):
    """Return recent generated messages from Supabase messages_log table."""
    sb    = get_supabase()
    query = sb.table("messages_log").select("*").order("created_at", desc=True)
    if segment:
        query = query.eq("segment", segment)
    try:
        resp = query.limit(limit).execute()
        return {"count": len(resp.data), "messages": resp.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase query failed: {e}")

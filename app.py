import time
import random
import pickle
import string

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield",
    page_icon="🛡️",
    layout="centered",
)

# ── NLTK setup (downloads once, cached across reruns) ────────────────────────
@st.cache_resource(show_spinner=False)
def ensure_nltk_ready():
    for res, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)
    return set(stopwords.words("english"))

STOP_WORDS = ensure_nltk_ready()
ps = PorterStemmer()


def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in STOP_WORDS and t not in string.punctuation]
    tokens = [ps.stem(t) for t in tokens]

    return " ".join(tokens)


# ── Load model + vectorizer ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error("⚠️ Could not find **model.pkl** or **vectorizer.pkl**. Make sure both files sit next to app.py.")
    st.stop()


# ── Dark glass dashboard CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

  :root {
    --bg-void: #090b14;
    --bg-void-2: #0e0a1c;
    --text-primary: #f4f5fa;
    --text-muted: #8d92a6;
    --glass: rgba(255,255,255,0.045);
    --glass-strong: rgba(255,255,255,0.075);
    --glass-border: rgba(255,255,255,0.10);
    --glass-hover-border: rgba(255,255,255,0.20);
    --accent-blue: #4f8cff;
    --accent-purple: #a855f7;
    --accent-pink: #ec4899;
    --accent-orange: #fb923c;
    --accent-safe: #2dd4a7;
    --accent-safe-glow: rgba(45,212,167,0.4);
    --accent-spam: #ff5577;
    --accent-spam-glow: rgba(255,85,119,0.4);
  }

  html, body, [class*="css"] {
    background: radial-gradient(ellipse 120% 80% at 50% -10%, var(--bg-void-2), var(--bg-void) 62%) fixed;
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
  }

  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1.2rem 1.4rem 4rem; max-width: 820px; position: relative; z-index: 1; }

  /* ── Ambient glow blobs (transform-only, GPU cheap) ── */
  .bg-blob { position: fixed; border-radius: 50%; filter: blur(90px); z-index: 0; pointer-events: none; will-change: transform; }
  .bg-blob-1 { width: 460px; height: 460px; top: -140px; left: -120px;
    background: radial-gradient(circle, var(--accent-blue) 0%, transparent 70%);
    opacity: 0.30; animation: drift1 24s ease-in-out infinite; }
  .bg-blob-2 { width: 420px; height: 420px; top: 10%; right: -160px;
    background: radial-gradient(circle, var(--accent-pink) 0%, transparent 70%);
    opacity: 0.28; animation: drift2 20s ease-in-out infinite; }
  .bg-blob-3 { width: 480px; height: 480px; bottom: -200px; left: 10%;
    background: radial-gradient(circle, var(--accent-orange) 0%, transparent 70%);
    opacity: 0.22; animation: drift3 28s ease-in-out infinite; }
  .bg-blob-4 { width: 360px; height: 360px; bottom: 5%; right: -100px;
    background: radial-gradient(circle, var(--accent-purple) 0%, transparent 70%);
    opacity: 0.24; animation: drift4 26s ease-in-out infinite; }
  @keyframes drift1 { 0%,100%{transform:translate(0,0) scale(1);} 50%{transform:translate(50px,40px) scale(1.15);} }
  @keyframes drift2 { 0%,100%{transform:translate(0,0) scale(1);} 50%{transform:translate(-40px,50px) scale(1.1);} }
  @keyframes drift3 { 0%,100%{transform:translate(0,0) scale(1);} 50%{transform:translate(40px,-40px) scale(1.2);} }
  @keyframes drift4 { 0%,100%{transform:translate(0,0) scale(1);} 50%{transform:translate(-30px,-30px) scale(1.12);} }

  /* ── Constellation network overlay ── */
  .bg-network { position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: 0.55; }
  .net-node { animation: nodePulse 4s ease-in-out infinite; }
  .net-node.n2 { animation-delay: 1.1s; }
  .net-node.n3 { animation-delay: 2.2s; }
  @keyframes nodePulse { 0%,100% { opacity: 0.5; } 50% { opacity: 1; } }

  /* ── Navbar ── */
  .navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.9rem 1.3rem; margin-bottom: 1.4rem;
    background: var(--glass);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid var(--glass-border);
    border-radius: 18px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
    animation: fadeInUp 0.5s cubic-bezier(.2,.8,.2,1) both;
  }
  .navbar-left { display: flex; align-items: center; gap: 0.8rem; }
  .logo-badge {
    width: 40px; height: 40px; border-radius: 12px; flex-shrink: 0;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple) 55%, var(--accent-pink));
    display: flex; align-items: center; justify-content: center; font-size: 1.15rem;
    box-shadow: 0 4px 18px rgba(168,85,247,0.45);
  }
  .navbar-logo-text { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.2rem; color: var(--text-primary); line-height: 1.1; }
  .navbar-tagline { font-size: 0.72rem; color: var(--text-muted); }
  .status-pill {
    display: flex; align-items: center; gap: 0.45rem;
    background: var(--glass-strong); border: 1px solid var(--glass-border);
    border-radius: 999px; padding: 0.4rem 0.85rem; font-size: 0.74rem; color: var(--text-muted);
  }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent-safe);
    box-shadow: 0 0 8px var(--accent-safe-glow); animation: statusPulse 2s infinite; }
  @keyframes statusPulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

  /* ── Stat cards ── */
  .stat-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 16px; padding: 1rem 1.1rem;
    backdrop-filter: blur(16px);
    transition: transform 0.3s cubic-bezier(.2,.8,.2,1), border-color 0.3s ease, box-shadow 0.3s ease;
    animation: fadeInUp 0.55s cubic-bezier(.2,.8,.2,1) both;
    height: 100%;
  }
  .stat-card:hover { transform: translateY(-4px); border-color: var(--glass-hover-border); box-shadow: 0 12px 30px rgba(0,0,0,0.35); }
  .stat-icon {
    width: 32px; height: 32px; border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-size: 0.95rem; margin-bottom: 0.6rem;
  }
  .stat-icon.blue { background: linear-gradient(135deg, var(--accent-blue), #86b3ff); }
  .stat-icon.spam { background: linear-gradient(135deg, var(--accent-spam), var(--accent-orange)); }
  .stat-icon.safe { background: linear-gradient(135deg, var(--accent-safe), #6ee6c4); }
  .stat-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; color: var(--text-primary); line-height: 1.1; }
  .stat-label { font-size: 0.72rem; color: var(--text-muted); margin-top: 0.2rem; }

  /* ── Hero / input panel ── */
  .hero-panel {
    background: var(--glass);
    backdrop-filter: blur(18px) saturate(180%);
    -webkit-backdrop-filter: blur(18px) saturate(180%);
    border: 1px solid var(--glass-border);
    border-radius: 18px;
    padding: 1.5rem 1.6rem 1.2rem;
    margin: 1.2rem 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
    animation: fadeInUp 0.6s cubic-bezier(.2,.8,.2,1) 0.05s both;
  }
  .hero-label {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 600;
    color: var(--text-primary); position: relative; display: inline-block; margin-bottom: 0.2rem;
  }
  .hero-label::after {
    content: ""; position: absolute; left: 0; bottom: -5px; width: 42px; height: 3px; border-radius: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-pink));
  }
  .hero-sub { font-size: 0.83rem; color: var(--text-muted); margin: 0.6rem 0 1rem; }

  .stTextArea textarea {
    background: rgba(255,255,255,0.035) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
  }
  .stTextArea textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(79,140,255,0.18) !important;
  }
  .stTextArea textarea::placeholder { color: #5c6178 !important; }

  /* ── Buttons ── */
  .stButton > button {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple) 60%, var(--accent-pink));
    color: #fff; font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.95rem;
    padding: 0.6rem 1.6rem; border: none; border-radius: 9px; cursor: pointer;
    box-shadow: 0 4px 18px rgba(168,85,247,0.35);
    transition: transform 0.25s cubic-bezier(.2,.8,.2,1), box-shadow 0.25s ease;
    will-change: transform;
  }
  .stButton > button::before {
    content: ""; position: absolute; top: 0; left: -60%; width: 40%; height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.4), transparent);
    transform: skewX(-20deg); transition: transform 0.6s ease;
  }
  .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 26px rgba(168,85,247,0.5); }
  .stButton > button:hover::before { transform: skewX(-20deg) translateX(340%); }
  .stButton > button:active { transform: translateY(0) scale(0.98); }

  /* secondary sample-chip buttons */
  div[data-testid="column"] .stButton > button {
    background: var(--glass-strong);
    color: var(--text-primary);
    border: 1px solid var(--glass-border);
    box-shadow: none;
    font-weight: 500;
    font-size: 0.82rem;
    padding: 0.45rem 1rem;
  }
  div[data-testid="column"] .stButton > button:hover {
    border-color: var(--accent-blue);
    box-shadow: 0 4px 16px rgba(79,140,255,0.25);
  }

  /* ── Scanning panel ── */
  .scan-panel {
    position: relative; overflow: hidden;
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 16px; padding: 2rem;
    text-align: center; margin-bottom: 1.2rem;
    animation: fadeInUp 0.35s ease both;
  }
  .scan-panel .scan-icon { font-size: 2rem; }
  .scan-line {
    position: absolute; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-pink), transparent);
    animation: scanMove 1.1s linear infinite;
  }
  @keyframes scanMove { 0% { top: 0%; } 100% { top: 100%; } }
  .scan-text { margin-top: 0.8rem; color: var(--text-muted); font-size: 0.85rem; letter-spacing: 0.3px; }

  /* ── Result card ── */
  .result-card {
    background: var(--glass-strong);
    backdrop-filter: blur(18px) saturate(180%);
    -webkit-backdrop-filter: blur(18px) saturate(180%);
    border-radius: 20px;
    padding: 2rem 1.8rem 1.6rem;
    text-align: center;
    margin-bottom: 1.2rem;
    animation: fadeInUp 0.45s cubic-bezier(.2,.8,.2,1) both;
  }
  .result-card.safe { border: 1px solid rgba(45,212,167,0.35); box-shadow: 0 16px 44px rgba(45,212,167,0.14), inset 0 1px 0 rgba(255,255,255,0.05); }
  .result-card.spam { border: 1px solid rgba(255,85,119,0.35); box-shadow: 0 16px 44px rgba(255,85,119,0.16), inset 0 1px 0 rgba(255,255,255,0.05); }

  .shield-badge {
    width: 68px; height: 68px; border-radius: 18px; margin: 0 auto 1rem;
    display: flex; align-items: center; justify-content: center; font-size: 2rem;
  }
  .shield-badge.safe { background: linear-gradient(135deg, var(--accent-safe), #6ee6c4); box-shadow: 0 10px 30px rgba(45,212,167,0.4); animation: shieldPopSafe 0.7s cubic-bezier(.34,1.56,.64,1) both; }
  .shield-badge.spam { background: linear-gradient(135deg, var(--accent-spam), var(--accent-orange)); box-shadow: 0 10px 30px rgba(255,85,119,0.4); animation: shieldPopSpam 0.7s cubic-bezier(.34,1.56,.64,1) both; }
  @keyframes shieldPopSafe {
    0% { transform: scale(0.4) rotate(-10deg); opacity: 0; }
    60% { transform: scale(1.15) rotate(4deg); opacity: 1; }
    100% { transform: scale(1) rotate(0); }
  }
  @keyframes shieldPopSpam {
    0% { transform: scale(0.4); opacity: 0; }
    45% { transform: scale(1.1) rotate(-8deg); opacity: 1; }
    60% { transform: rotate(8deg); }
    75% { transform: rotate(-5deg); }
    90% { transform: rotate(3deg); }
    100% { transform: rotate(0); }
  }

  .verdict-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; letter-spacing: 0.3px; margin-bottom: 0.3rem; }
  .verdict-title.safe { color: var(--accent-safe); }
  .verdict-title.spam { color: var(--accent-spam); }
  .verdict-sub { color: var(--text-muted); font-size: 0.87rem; margin-bottom: 1.2rem; }

  .conf-label { display: flex; justify-content: space-between; font-size: 0.76rem; color: var(--text-muted); margin-bottom: 0.35rem; }
  .conf-track { position: relative; width: 100%; height: 11px; border-radius: 8px; background: rgba(255,255,255,0.07); overflow: hidden; }
  .conf-fill { position: absolute; inset: 0; border-radius: 8px; transform-origin: left center; transform: scaleX(0);
    animation: fillBar 0.9s 0.15s cubic-bezier(.2,.8,.2,1) forwards; }
  .conf-fill.safe { background: linear-gradient(90deg, var(--accent-safe), #6ee6c4); box-shadow: 0 0 12px var(--accent-safe-glow); }
  .conf-fill.spam { background: linear-gradient(90deg, var(--accent-spam), var(--accent-orange)); box-shadow: 0 0 12px var(--accent-spam-glow); }
  @keyframes fillBar { to { transform: scaleX(var(--target-scale)); } }

  /* ── History rows ── */
  .history-title { font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; font-weight: 600; color: var(--text-primary); margin: 1.6rem 0 0.6rem; }
  .history-chip {
    display: flex; align-items: center; gap: 0.6rem;
    background: var(--glass); border: 1px solid var(--glass-border);
    border-radius: 12px; padding: 0.6rem 0.85rem; margin-bottom: 0.5rem;
    font-size: 0.82rem; color: var(--text-muted);
    opacity: 0; animation: fadeInUp 0.4s ease forwards;
    transition: transform 0.25s ease, border-color 0.25s ease;
  }
  .history-chip:hover { transform: translateX(4px); border-color: var(--glass-hover-border); }
  .history-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .history-dot.safe { background: var(--accent-safe); box-shadow: 0 0 8px var(--accent-safe-glow); }
  .history-dot.spam { background: var(--accent-spam); box-shadow: 0 0 8px var(--accent-spam-glow); }
  .history-text { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text-primary); }
  .history-pill { font-size: 0.7rem; font-weight: 600; padding: 3px 10px; border-radius: 999px; }
  .history-pill.safe { background: rgba(45,212,167,0.15); color: var(--accent-safe); }
  .history-pill.spam { background: rgba(255,85,119,0.15); color: var(--accent-spam); }

  @keyframes fadeInUp { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: translateY(0); } }

  .app-footer { text-align: center; color: #565b70; font-size: 0.75rem; padding-top: 2.5rem; letter-spacing: 0.3px; }

  @media (prefers-reduced-motion: reduce) {
    .bg-blob, .net-node, .navbar, .stat-card, .hero-panel, .result-card, .shield-badge,
    .conf-fill, .history-chip, .scan-line, .status-dot, .stButton > button {
      animation: none !important; transition: none !important;
    }
  }
</style>

<div class="bg-blob bg-blob-1"></div>
<div class="bg-blob bg-blob-2"></div>
<div class="bg-blob bg-blob-3"></div>
<div class="bg-blob bg-blob-4"></div>

<svg class="bg-network" viewBox="0 0 1000 1000" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#4f8cff"/>
      <stop offset="50%" stop-color="#a855f7"/>
      <stop offset="100%" stop-color="#ec4899"/>
    </linearGradient>
    <filter id="nodeGlow" x="-200%" y="-200%" width="500%" height="500%">
      <feGaussianBlur stdDeviation="8" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <g stroke="url(#lineGrad)" stroke-width="1" opacity="0.25">
    <line x1="120" y1="140" x2="380" y2="260"/>
    <line x1="380" y1="260" x2="700" y2="180"/>
    <line x1="700" y1="180" x2="880" y2="360"/>
    <line x1="380" y1="260" x2="520" y2="520"/>
    <line x1="520" y1="520" x2="220" y2="640"/>
    <line x1="520" y1="520" x2="820" y2="700"/>
    <line x1="820" y1="700" x2="920" y2="880"/>
    <line x1="220" y1="640" x2="120" y2="860"/>
  </g>
  <g fill="#fff">
    <circle class="net-node n1" cx="120" cy="140" r="3.5" fill="#4f8cff" filter="url(#nodeGlow)"/>
    <circle class="net-node n2" cx="380" cy="260" r="3" fill="#a855f7" filter="url(#nodeGlow)"/>
    <circle class="net-node n3" cx="700" cy="180" r="3.5" fill="#ec4899" filter="url(#nodeGlow)"/>
    <circle class="net-node n1" cx="880" cy="360" r="3" fill="#fb923c" filter="url(#nodeGlow)"/>
    <circle class="net-node n2" cx="520" cy="520" r="3.5" fill="#4f8cff" filter="url(#nodeGlow)"/>
    <circle class="net-node n3" cx="220" cy="640" r="3" fill="#2dd4a7" filter="url(#nodeGlow)"/>
    <circle class="net-node n1" cx="820" cy="700" r="3.5" fill="#a855f7" filter="url(#nodeGlow)"/>
    <circle class="net-node n2" cx="920" cy="880" r="3" fill="#ec4899" filter="url(#nodeGlow)"/>
    <circle class="net-node n3" cx="120" cy="860" r="3.5" fill="#fb923c" filter="url(#nodeGlow)"/>
  </g>
</svg>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "msg_input" not in st.session_state:
    st.session_state.msg_input = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


SPAM_EXAMPLES = [
    "Congratulations! You've WON a $1000 Walmart gift card. Click here now to claim your prize!!!",
    "URGENT: Your account has been suspended. Verify your details immediately at bit.ly/verify-now to avoid closure.",
    "You have been selected to receive a FREE iPhone 15! Text CLAIM to 88900 before it expires today.",
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward. Call 09061701461.",
    "Your loan of $5000 has been approved! No credit check needed. Reply YES to receive funds within 24 hours.",
    "FreeMsg: Txt CHAT to 86688 now and get a FREE 100 text bundle plus enter our £5000 cash draw!",
    "Final notice: Your package could not be delivered. Confirm your address here to reschedule: http://del-track.co/x9",
    "Congrats! You've been chosen for a $500 Amazon voucher. Claim now, offer expires in 15 minutes!",
]

NORMAL_EXAMPLES = [
    "Hey, are we still on for lunch tomorrow at 1pm?",
    "Can you send me the notes from today's meeting when you get a chance?",
    "Running about 10 minutes late, see you soon!",
    "Happy birthday! Hope you have an amazing day, let's catch up this weekend.",
    "Don't forget to pick up milk on your way home.",
    "The flight got delayed by an hour, new arrival time is 6:45pm.",
    "Thanks for helping me move last weekend, I owe you dinner.",
    "Reminder: dentist appointment on Thursday at 3pm.",
]


def set_sample(pool):
    st.session_state.msg_input = random.choice(pool)


total_checked = len(st.session_state.history)
spam_count = sum(1 for h in st.session_state.history if h["spam"])
safe_rate = round(100 * (total_checked - spam_count) / total_checked, 0) if total_checked else 100


# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="navbar-left">
    <div class="logo-badge">🛡️</div>
    <div>
      <div class="navbar-logo-text">SpamShield</div>
      <div class="navbar-tagline">ML-powered SMS &amp; email spam detection</div>
    </div>
  </div>
  <div class="status-pill"><span class="status-dot"></span> Model active</div>
</div>
""", unsafe_allow_html=True)


# ── Stat row ──────────────────────────────────────────────────────────────────
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-icon blue">🔎</div>
      <div class="stat-value">{total_checked}</div>
      <div class="stat-label">Messages checked</div>
    </div>
    """, unsafe_allow_html=True)
with s2:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-icon spam">🚨</div>
      <div class="stat-value">{spam_count}</div>
      <div class="stat-label">Spam blocked</div>
    </div>
    """, unsafe_allow_html=True)
with s3:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-icon safe">✅</div>
      <div class="stat-value">{safe_rate:.0f}%</div>
      <div class="stat-label">Safe rate</div>
    </div>
    """, unsafe_allow_html=True)


# ── Hero input panel ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-panel">
  <div class="hero-label">Check a message</div>
  <div class="hero-sub">Paste any SMS or email text below and we'll scan it for spam signals.</div>
</div>
""", unsafe_allow_html=True)

message = st.text_area(
    "Message",
    key="msg_input",
    height=140,
    label_visibility="collapsed",
    placeholder="Paste an SMS or email message here...",
)

c1, c2, c3 = st.columns([1, 1, 1.3])
with c1:
    st.button(
        "🎲 Try spam example",
        on_click=set_sample,
        args=(SPAM_EXAMPLES,),
        use_container_width=True,
    )
with c2:
    st.button(
        "🎲 Try normal example",
        on_click=set_sample,
        args=(NORMAL_EXAMPLES,),
        use_container_width=True,
    )
with c3:
    analyze_clicked = st.button("🔍 Analyze Message", use_container_width=True)

result_area = st.empty()

if analyze_clicked:
    if not message.strip():
        st.warning("Please enter a message to analyze.")
    else:
        result_area.markdown("""
        <div class="scan-panel">
          <div class="scan-line"></div>
          <div class="scan-icon">🛡️</div>
          <div class="scan-text">Scanning message for spam signals…</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.7)

        cleaned = transform_text(message)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        spam_prob = float(model.predict_proba(vector)[0][1]) * 100
        is_spam = bool(prediction == 1)

        st.session_state.last_result = {"is_spam": is_spam, "spam_prob": spam_prob}
        st.session_state.history.insert(0, {
            "text": message.strip()[:60],
            "spam": is_spam,
        })
        st.session_state.history = st.session_state.history[:5]
        st.rerun()

if st.session_state.last_result is not None:
    is_spam = st.session_state.last_result["is_spam"]
    spam_prob = st.session_state.last_result["spam_prob"]
    css_class = "spam" if is_spam else "safe"
    verdict = "SPAM DETECTED" if is_spam else "LOOKS SAFE"
    subtitle = (
        "This message shows strong spam signals — links, urgency, or prize language."
        if is_spam
        else "No notable spam signals found in this message."
    )
    shield_icon = "🚨" if is_spam else "✅"
    target_scale = round(spam_prob if is_spam else (100 - spam_prob), 1) / 100

    result_area.markdown(f"""
    <div class="result-card {css_class}">
      <div class="shield-badge {css_class}">{shield_icon}</div>
      <div class="verdict-title {css_class}">{verdict}</div>
      <div class="verdict-sub">{subtitle}</div>
      <div class="conf-label">
        <span>Confidence</span>
        <span>{round(target_scale * 100, 1)}%</span>
      </div>
      <div class="conf-track">
        <div class="conf-fill {css_class}" style="--target-scale:{target_scale};"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.history:
    st.markdown('<div class="history-title">Recent checks</div>', unsafe_allow_html=True)
    chips = ""
    for i, item in enumerate(st.session_state.history):
        dot_class = "spam" if item["spam"] else "safe"
        label = "Spam" if item["spam"] else "Safe"
        delay = 0.05 * i
        chips += f"""
        <div class="history-chip" style="animation-delay:{delay}s;">
          <span class="history-dot {dot_class}"></span>
          <span class="history-text">{item['text']}</span>
          <span class="history-pill {dot_class}">{label}</span>
        </div>
        """
    st.markdown(chips, unsafe_allow_html=True)

st.markdown('<div class="app-footer">SpamShield · TF-IDF + Naive Bayes · Made with Streamlit</div>', unsafe_allow_html=True)
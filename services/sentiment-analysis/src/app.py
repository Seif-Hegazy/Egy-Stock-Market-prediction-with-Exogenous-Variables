import sys
import os

# --- Runtime Dependency Hotfix ---
LIB_DIR = "/app/src/libs"
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

import streamlit as st
import ollama
import yfinance as yf
import importlib

# Force reload to ensure we get the version from /app/src/libs
try:
    importlib.reload(yf)
except Exception as e:
    print(f"Reload failed: {e}")

import plotly.graph_objects as go
import json
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import requests
from bs4 import BeautifulSoup

# --- Page Config ---
st.set_page_config(
    page_title="EgySentiment | Financial Intelligence",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Debug Info in Sidebar (Hidden) ---
# st.sidebar.markdown("### 🐞 Debug Info")
# st.sidebar.code(f"YF Version: {yf.__version__}")
# st.sidebar.code(f"YF Path: {yf.__file__}")
# st.sidebar.code(f"Sys Path[0]: {sys.path[0]}")

# --- Custom CSS (Modern Dark Theme) ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Card Style */
    .metric-card {
        background: linear-gradient(145deg, #1E1E1E, #252525);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #444;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    .big-font {
        font-size: 56px !important;
        font-weight: 800;
        margin: 10px 0;
        background: -webkit-linear-gradient(45deg, #eee, #999);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sentiment Colors */
    .sent-positive { color: #00CC96 !important; -webkit-text-fill-color: #00CC96 !important; }
    .sent-negative { color: #EF553B !important; -webkit-text-fill-color: #EF553B !important; }
    .sent-neutral { color: #AB63FA !important; -webkit-text-fill-color: #AB63FA !important; }
    
    /* Input Area */
    .stTextArea textarea {
        background-color: #151920;
        border: 1px solid #333;
        border-radius: 12px;
        font-size: 16px;
        color: #eee;
    }
    .stTextArea textarea:focus {
        border-color: #00CC96;
        box-shadow: 0 0 0 1px #00CC96;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #00CC96, #00A87E);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# --- Comprehensive Stock Mapping (EGX 30) ---
# Format: "Display Name": {"ticker": "TICKER.CA", "keywords": ["list", "of", "keywords"]}
STOCK_DATA = {
    # --- Banking Sector ---
    "Commercial International Bank (CIB)": {
        "ticker": "COMI.CA",
        "keywords": ["CIB", "COMI", "Commercial International Bank", "البنك التجاري الدولي", "التجاري الدولي", "CIB Egypt", 
                    "CIB Capital", "CIB Corporate", " CIB Retail", "Hisham Ezz Al-Arab", "CIB credit", "CIB loan",
                    "CIB SME", "CIB digital", "CIB fintech", "CIB branch", "CIB deposits", "CIB NPL",
                    "commercial international", "largest private bank egypt", "premier bank"]
    },
    # "QNB Alahli": {
    #     "ticker": "QNBA.CA",
    #     "keywords": [" QNB", "QNBA", "Qatar National Bank", "بنك قطر الوطني", "قطر الوطني", "QNB Alahli", "بنك قطر الوطني الأهلي",
    #                 "QNB Egypt", "QNB branch", "QNB digital", "QNB corporate", "QNB retail", "national bank of qatar egypt",
    #                 "qnb loan", "qnb credit", "qnb mortgage", "qnb deposit"]
    # },
    "Crédit Agricole Egypt": {
        "ticker": "CIEB.CA",
        "keywords": ["Credit Agricole", "CIEB", "Crédit Agricole", "كريدي أجريكول", "بنك كريدي أجريكول",
                    "CAE", "Credit Agricole Egypt", "CA Egypt", "french bank egypt", "credit agricole branch",
                    "credit agricole corporate", "credit agricole retail", "CA loan"]
    },
    "Housing & Development Bank": {
        "ticker": "HDBK.CA",
        "keywords": ["HDBK", "Housing & Development Bank", "Housing and Development Bank", "بنك التعمير والإسكان", "التعمير والإسكان",
                    "HDB", "housing bank", "development bank", "mortgage bank", "real estate financing",
                    "hdb loan", "hdb mortgage", "housing finance", "construction loan"]
    },
    "Faisal Islamic Bank of Egypt": {
        "ticker": "FAIT.CA",
        "keywords": ["Faisal Islamic Bank", "FAIT", "FAITA", "بنك فيصل الإسلامي", "فيصل الإسلامي",
                    "FIB", "islamic bank", "sharia banking", "faisal bank", "islamic finance egypt",
                    "murabaha", "sukuk", "islamic banking egypt", "sharia compliant"]
    },
    "Abu Dhabi Islamic Bank (ADIB)": {
        "ticker": "ADIB.CA",
        "keywords": ["ADIB", "Abu Dhabi Islamic Bank", "مصرف أبوظبي الإسلامي", "أبوظبي الإسلامي", "ADIB Egypt",
                    "ADIB branch", "ADIB digital", "abu dhabi bank egypt", "uae bank egypt"]
    },
    "Al Baraka Bank Egypt": {
        "ticker": "SAUD.CA",
        "keywords": ["Al Baraka", "SAUD", "Al Baraka Bank", "بنك البركة", "البركة مصر",
                    "Baraka Bank", "Islamic bank", "al baraka egypt"]
    },
    "Egyptian Gulf Bank (EGBANK)": {
        "ticker": "EGBE.CA",
        "keywords": ["EGBANK", "EGBE", "Egyptian Gulf Bank", "البنك المصري الخليجي", "المصري الخليجي",
                    "EG Bank", "egyptian gulf", "egb"]
    },
    "Export Development Bank of Egypt (EBank)": {
        "ticker": "EXPA.CA",
        "keywords": ["EBank", "EXPA", "Export Development Bank", "البنك المصري لتنمية الصادرات", "تنمية الصادرات",
                    "export bank", "edb egypt", "export financing", "trade finance"]
    },

    # --- Non-Bank Financial Services ---
    "EFG Hermes": {
        "ticker": "HRHO.CA",
        "keywords": ["EFG Hermes", "HRHO", "EFG", "EFG Holding", "المجموعة المالية هيرميس", "هيرميس", "هيرميس القابضة",
                    "EFG investment bank", "Hermes capital", "EFG brokerage", "EFG finance", "karim awad",
                    "Mohamed Ebeid", "frontier markets", "emerging markets bank", "MENA investment",
                    "EFG asset management", "Hermes Fund", "EFG M&A", "beltone acquisition"]
    },
    "E-Finance": {
        "ticker": "EFIH.CA",
        "keywords": ["E-Finance", "EFIH", "e-finance", "إي فاينانس", "اي فاينانس", "e-finance for Digital and Financial Investments",
                    "digital payment", "government payment", "electronic payment", "fintech egypt",
                    "e-payment", "digital transactions", "government services"]
    },
    "Fawry": {
        "ticker": "FWRY.CA",
        "keywords": ["Fawry", "FWRY", "Fawry for Banking Technology", "فوري", "شركة فوري", "فوري للمدفوعات",
                    "payment gateway", "digital wallet", "bill payment", "mobile payment", "fintech",
                    "fawry pay", "fawry plus", "cashless", "e-wallet", "Ashraf Sabry",
                    "fawry merchant", "fawry kiosk", "payment solutions"]
    },
    "Beltone Financial": {
        "ticker": "BTFH.CA",
        "keywords": ["Beltone", "BTFH", "Beltone Financial", "بلتون", "بلتون المالية", "بلتون القابضة",
                    "beltone", "investment bank", "asset management", "brokerage"]
    },
    "CI Capital": {
        "ticker": "CICH.CA",
        "keywords": ["CI Capital", "CICH", "سي آي كابيتال", "سي اي كابيتال",
                    "investment bank", "asset management", "commercial international capital"]
    },

    # --- Real Estate & Construction ---
    "Talaat Moustafa Group (TMG)": {
        "ticker": "TMGH.CA",
        "keywords": ["Talaat Moustafa", "TMGH", "TMG", "TMG Holding", "طلعت مصطفى", "مجموعة طلعت مصطفى", 
                    "Madinaty", "Rehab City", "مدينتي", "الرحاب", "Al Rehab", "Celia",
                    "talaat moustafa development", "TMG projects", "new cairo developer",
                    "Hisham Talaat Moustafa", "egypt real estate", "largest developer",
                    "Noor", "May Fair", "residential compound", "commercial mall"]
    },
    "Palm Hills Developments": {
        "ticker": "PHDC.CA",
        "keywords": ["Palm Hills", "PHDC", "Palm Hills Developments", "بالم هيلز", "بالم هيلز للتعمير", "Badya", "بادية",
                    "Palm parks", "Palm valley", "6 october", "october palm", "real estate developer",
                    "residential project", "Mansour Hamed", "palm hills delivery"]
    },
    "Sixth of October Development & Investment (SODIC)": {
        "ticker": "OCDI.CA",
        "keywords": ["SODIC", "OCDI", "Sixth of October Development", "سوديك", "السادس من أكتوبر للتنمية",
                    "Sheikh Zayed", "West Cairo", "Allegria", "Eastown", "Villette", "The Polygon",
                    "october city", "sodic project", "Magued Sherif", "sodic delivery", "mixed-use"]
    },
    "Madinet Masr (MNHD)": {
        "ticker": "MASR.CA",
        "keywords": ["Madinet Masr", "MASR", "Madinet Nasr", "MNHD", "مدينة مصر", "مدينة نصر للإسكان", 
                    "Taj City", "تاج سيتي", "Sarai", "سراي", "Fifth Square", "خامس سكوير",
                    "New Cairo developer", "Abdallah Sallam", "madinet nasr housing"]
    },
    "Heliopolis Housing": {
        "ticker": "HELI.CA",
        "keywords": ["Heliopolis", "HELI", "Heliopolis Company for Housing", "مصر الجديدة", "مصر الجديدة للإسكان", "مصر الجديدة للاسكان والتعمير",
                    "heliopolis club", "heliopolis hospital", "sheraton buildings", "cairo heliopolis",
                    "oldest developer egypt", "historic developer"]
    },
    "Orascom Construction": {
        "ticker": "ORAS.CA",
        "keywords": ["Orascom Construction", "ORAS", "Orascom", "أوراسكوم للإنشاءات", "أوراسكوم كونستراكشون",
                    "OC", "construction egypt", "infrastructure", "Osama Bishai", "Besix",
                    "orascom projects", "engineering", "contractor", "EPC"]
    },
    "Emaar Misr": {
        "ticker": "EMFD.CA",
        "keywords": ["Emaar", "EMFD", "Emaar Misr", "إعمار", "إعمار مصر", "Marassi", "مراسي",
                    "Uptown Cairo", "Mivida", "ميفيدا", "emaar egypt", "UAE developer egypt",
                    "North Coast", "Emaar Properties", "emaar delivery"]
    },

    # --- Industrial & Basic Resources ---
    "Elsewedy Electric": {
        "ticker": "SWDY.CA",
        "keywords": ["Elsewedy", "SWDY", "El Sewedy", "Elsewedy Electric", "السويدي", "السويدي إليكتريك", "السويدي للكابلات",
                    "cables", "transformers", "electrical equipment", "Ahmed El Sewedy",
                    "elsewedy cables", "power", "energy infrastructure", "meters", "wind energy",
                    "elsewedy solar", "elsewedy projects", "paints"]
    },
    # "Ezz Steel": {
    #     "ticker": "ESRS.CA",
    #     "keywords": ["Ezz Steel", "ESRS", "Ezz", "Al Ezz Dekheila", "حديد عز", "عز الدخيلة", "مجموعة عز",
    #                 "steel egypt", "rebar", "steel products", "iron", "ezz ahmed",
    #                 "flat steel", "long steel", "construction steel", "monopoly steel"]
    # },
    "Abu Qir Fertilizers": {
        "ticker": "ABUK.CA",
        "keywords": ["Abu Qir", "ABUK", "Abu Qir Fertilizers", "أبو قير", "أبو قير للأسمدة", "ابوقير",
                    "fertilizers", "urea", "ammonia", "agricultural", "Abu Kir",
                    "nitrogen fertilizer", "chemical fertilizer", "agriculture egypt"]
    },
    "Misr Fertilizers Production (MOPCO)": {
        "ticker": "MFPC.CA",
        "keywords": ["MOPCO", "MFPC", "Misr Fertilizers", "موبكو", "مصر لإنتاج الأسمدة",
                    "fertilizers damietta", "mopco plant", "nitrogen", "agricultural inputs"]
    },
    "Sidi Kerir Petrochemicals (SIDPEC)": {
        "ticker": "SKPC.CA",
        "keywords": ["Sidi Kerir", "SKPC", "Sidpec", "سيدي كرير", "سيدبك", "سيدي كرير للبتروكيماويات",
                    "petrochemicals", "polyethylene", "plastics", "chemicals", "sidpec plant",
                    "alexandria petrochemical"]
    },
    "Alexandria Mineral Oils (AMOC)": {
        "ticker": "AMOC.CA",
        "keywords": ["AMOC", "Alexandria Mineral Oils", "أموك", "زيوت معدنية", "الاسكندرية للزيوت المعدنية",
                    "lubricants", "base oils", "mineral oils", "amoc refinery"]
    },
    # "Kima": {
    #     "ticker": "KIMA.CA",
    #     "keywords": ["Kima", "KIMA", "Egyptian Chemical Industries", "كيما", "الصناعات الكيماوية المصرية",
    #                 "chemicals aswan", "fertilizers kima", "ammonia kima"]
    # },

    # --- Telecom & Technology ---
    "Telecom Egypt (WE)": {
        "ticker": "ETEL.CA",
        "keywords": ["Telecom Egypt", "ETEL", "WE", "TE", "المصرية للاتصالات", "وي", "تي إي داتا",
                    "landline", "ADSL", "fiber", "internet", "data", "TE Data",
                    "telecom monopoly", "international gateway", "4G license", "mobile license",
                    "Adel Hamed", "telecom infrastructure"]
    },

    # --- Consumer & Healthcare ---
    "Eastern Company": {
        "ticker": "EAST.CA",
        "keywords": ["Eastern Company", "EAST", "Eastern Tobacco", "الشرقية للدخان", "ايسترن كومباني", "سجائر",
                    "cigarettes", "tobacco", "cleopatra cigarettes", "marlboro egypt",
                    "philip morris egypt", "smoking", "tax revenue", "eastern tobacco monopoly"]
    },
    "Juhayna Food Industries": {
        "ticker": "JUFO.CA",
        "keywords": ["Juhayna", "JUFO", "جهينة", "جهينه", "جهينة للصناعات الغذائية",
                    "dairy", "milk", "juice", "yogurt", "cheese", "beyti", "juhayna products",
                    "food industry", "Safwan Thabet", "juhayna milk"]
    },
    "Edita Food Industries": {
        "ticker": "EFID.CA",
        "keywords": ["Edita", "EFID", "إيديتا", "ايديتا", "إيديتا للصناعات الغذائية",
                    "Todo", "Molto", "Freska", "HoHos", "MiMix", "snacks", "biscuits",
                    "food manufacturer", "Hani Berzi", "edita export", "sweets"]
    },
    "Ibnsina Pharma": {
        "ticker": "ISPH.CA",
        "keywords": ["Ibnsina", "ISPH", "Ibnsina Pharma", "ابن سينا", "ابن سينا فارما",
                    "pharmaceutical distribution", "drug distributor", "pharmacy", "medicines",
                    "healthcare distribution", "medical supplies", "Mohsen El Mahdy"]
    },
    "Cleopatra Hospitals": {
        "ticker": "CLHO.CA",
        "keywords": ["Cleopatra", "CLHO", "Cleopatra Hospitals Group", "CHG", "مستشفيات كليوباترا", "مجموعة كليوباترا",
                    "hospitals", "healthcare", "medical services", "private hospital",
                    "cleopatra medical", "hospital chain", "Ahmed Ezzeldin"]
    },
    "GB Corp (Ghabbour)": {
        "ticker": "GBCO.CA",
        "keywords": ["GB Corp", "GBCO", "GB Auto", "Ghabbour", "جي بي أوتو", "غبور", "جي بي كورب",
                    "Hyundai Egypt", "Mazda Egypt", "automotive", "cars", "vehicles",
                    "Geely Egypt", "Chery", "Raouf Ghabbour", "auto distributor",
                    "car sales", "passenger vehicles", "commercial vehicles"]
    },

    # --- Others ---
    "Egypt Kuwait Holding": {
        "ticker": "EKHO.CA",
        "keywords": ["Egypt Kuwait Holding", "EKHO", "EKH", "القابضة المصرية الكويتية", "المصرية الكويتية",
                    "investment holding", "manufacturing", "ceramic", "National Cement",
                    "Qurain Petrochemicals", "ekh subsidiaries"]
    },
    "Qalaa Holdings": {
        "ticker": "CCAP.CA",
        "keywords": ["Qalaa", "CCAP", "Citadel Capital", "القلعة", "القلعة للاستشارات المالية",
                    "Taqa Arabia", "energy", "infrastructure investment", "Ahmed Heikal",
                    "private equity egypt", "qalaa energy"]
    },
    "Egyptian Satellites (NileSat)": {
        "ticker": "EGSA.CA",
        "keywords": ["NileSat", "EGSA", "Egyptian Satellites", "نايل سات", "المصرية للأقمار الصناعية",
                    "satellite", "broadcasting", "telecommunications satellite", "nilesat frequency",
                    "tv broadcasting"]
    }
}

# --- Helper Functions ---
def analyze_text(text):
    try:
        # Groq Client Setup
        from groq import Groq
        import os
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "neutral", "Error: GROQ_API_KEY not found in environment variables."
            
        client = Groq(api_key=api_key)
        
        prompt = f"""Analyze the sentiment of this text regarding Egyptian finance.
        
        Text: {text[:2000]}
        
        Respond ONLY with valid JSON in this exact format:
        {{"sentiment": "positive/negative/neutral", "reasoning": "brief explanation"}}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        
        result = json.loads(content)
        return result.get("sentiment", "neutral").lower(), result.get("reasoning", "")
        
    except Exception as e:
        return "neutral", f"Error: {str(e)}"

def generate_market_report(articles, stock_name="General Market", sentiment_scores=None):
    """
    Generates a professional executive summary from articles.
    
    Args:
        articles: List of article texts
        stock_name: Name of the stock/market (e.g., "CIB", "General Market")
        sentiment_scores: Optional list of sentiment scores (-1, 0, 1) for each article
    
    Returns:
        Formatted executive summary as markdown string
    """
    # Context Management: Limit to top 50 most recent articles to fit context window
    MAX_ARTICLES = 50
    if len(articles) > MAX_ARTICLES:
        articles = articles[:MAX_ARTICLES]
        if sentiment_scores:
            sentiment_scores = sentiment_scores[:MAX_ARTICLES]
    
    # Calculate sentiment distribution if provided
    sentiment_context = ""
    if sentiment_scores:
        positive_pct = (sentiment_scores.count(1) / len(sentiment_scores) * 100) if sentiment_scores else 0
        negative_pct = (sentiment_scores.count(-1) / len(sentiment_scores) * 100) if sentiment_scores else 0
        neutral_pct = (sentiment_scores.count(0) / len(sentiment_scores) * 100) if sentiment_scores else 0
        
        sentiment_context = f"""
Sentiment Distribution:
- Positive: {positive_pct:.1f}%
- Negative: {negative_pct:.1f}%
- Neutral: {neutral_pct:.1f}%
"""
    
    combined_text = "\n\n".join([f"- {a[:500]}" for a in articles])  # Limit each article to 500 chars
    
    prompt = f"""You are a Senior Financial Analyst preparing an executive briefing for institutional investors.

**Assignment:** Analyze the following news coverage for {stock_name} and create a concise but insightful Executive Summary.

{sentiment_context}

**News Headlines & Excerpts:**
{combined_text}

**IMPORTANT: Your response MUST be in MARKDOWN format with clear headers. DO NOT use JSON format.**

**Your Executive Summary should include these 5 sections:**

### Market Sentiment
[Overall tone: bullish/bearish/mixed with 1-2 sentence justification]

### Key Drivers
[List 2-3 main factors as bullet points]

### Risks
[Any downside concerns as bullet points]

### Opportunities
[Growth catalysts as bullet points]

### Outlook
[Brief forward-looking statement, 1-2 sentences]

**Writing Guidelines:**
- Use markdown headers (###) for each section
- Maximum 250 words total
- Professional tone suitable for C-suite executives
- Focus on actionable insights, not just descriptions
- BE SPECIFIC - avoid vague statements

Write your analysis in MARKDOWN format with the sections above:"""

    try:
        # Groq Client Setup
        from groq import Groq
        import os
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "**Error**: GROQ_API_KEY not found. Please check your environment configuration."
            
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {'role': 'system', 'content': 'You are an expert financial analyst with 15+ years of experience covering emerging markets. Your reports are known for their clarity, depth, and actionable insights.'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.3
        )
        
        summary = response.choices[0].message.content
        
        # Add metadata footer
        footer = f"\n\n---\n*Analysis based on {len(articles)} recent articles | Generated: {time.strftime('%Y-%m-%d %H:%M')} | Powered by Groq Llama 3*"
        
        return summary + footer
        
    except Exception as e:
        return f"""**Error Generating Report**

An error occurred while generating the executive summary: {str(e)}

**Troubleshooting:**
- Verify GROQ_API_KEY is set
- Check internet connection
- Verify article data is properly formatted

*Contact support if error persists.*"""

def load_today_articles(data_file="/app/data/news/testing_data.jsonl", days_back=1):
    """
    Load articles from the data file for today (or last N days).
    
    Args:
        data_file: Path to the JSONL data file
        days_back: Number of days to look back (default: 1 = today only)
    
    Returns:
        DataFrame with today's articles, or None if file doesn't exist
    """
    if not os.path.exists(data_file):
        return None
    
    # Calculate date threshold
    today = datetime.now().date()
    date_threshold = today - timedelta(days=days_back-1)
    
    articles = []
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line.strip())
                    
                    # Try to parse the date from the article
                    article_date = None
                    if 'date' in article:
                        try:
                            # Handle various date formats
                            date_str = article['date']
                            if isinstance(date_str, str):
                                # Try YYYY-MM-DD format first
                                if 'T' in date_str or ' ' in date_str:
                                    article_date = datetime.fromisoformat(date_str.split('T')[0]).date()
                                else:
                                    article_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        except:
                            pass
                    
                    # If date parsing failed, try timestamp
                    if not article_date and 'timestamp' in article:
                        try:
                            article_date = datetime.fromisoformat(article['timestamp']).date()
                        except:
                            pass
                    
                    # Include article if it's within our date range (or no date available)
                    if not article_date or article_date >= date_threshold:
                        articles.append(article)
                    
                except (json.JSONDecodeError, ValueError):
                    continue
        
        if articles:
            return pd.DataFrame(articles)
        return None
        
    except Exception as e:
        st.error(f"Error loading articles: {e}")
        return None

def get_sentiment_score(sentiment):
    if sentiment == "positive": return 1
    if sentiment == "negative": return -1
    return 0

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/pyramids.png", width=64)
    st.title("EgySentiment")
    st.caption("v1.3 | Local Inference Engine")
    
    st.markdown("---")
    st.subheader("📈 Market Context")
    selected_name = st.selectbox(
        "Select Company",
        options=list(STOCK_DATA.keys()),
        index=0
    )
    selected_ticker = STOCK_DATA[selected_name]["ticker"]
    st.caption(f"Ticker: **{selected_ticker}**")
    
    st.markdown("---")
    st.info("💡 **Tip:** Use the 'Batch Processing' tab to generate features for your forecasting model.")

# --- Main Content ---
st.markdown("## 🦅 Financial Intelligence Dashboard")

tab1, tab2, tab3 = st.tabs(["📅 Today's Market Digest", "⚡ Live Analysis", "🏭 Batch Processing"])

# === TAB 1: TODAY'S MARKET DIGEST ===
with tab1:
    st.markdown("### 📅 Today's Market Intelligence Summary")
    st.caption("Automatically analyze today's collected financial news. No file upload required - we'll load the latest articles for you!")
    
    # Configuration
    col1, col2 = st.columns([2, 1])
    with col1:
        days_back = st.slider("📆 Days to Include", min_value=1, max_value=7, value=1, 
                              help="1 = Today only, 7 = Last week")
    with col2:
        stock_filter = st.selectbox("🎯 Focus Stock (Optional)", 
                                    ["All Stocks"] + list(STOCK_DATA.keys()),
                                    help="Filter analysis to specific stock")
    
    if st.button("🔄 Load & Analyze Today's News", type="primary", use_container_width=True):
        with st.spinner(f"Loading articles from last {days_back} day(s)..."):
            # Load today's articles
            df_today = load_today_articles(days_back=days_back)
            
            if df_today is None or df_today.empty:
                st.warning(f"📭 No articles found for the last {days_back} day(s). The data collection pipeline may not have run yet today.")
                st.info("💡 **Tip:** The Airflow DAG runs every 4 hours. Check back later or adjust the 'Days to Include' slider.")
            else:
                st.success(f"✅ Loaded {len(df_today)} articles from the last {days_back} day(s)!")
                
                # Apply stock filter if selected
                if stock_filter != "All Stocks":
                    keywords = STOCK_DATA[stock_filter]["keywords"]
                    keyword_list = [k.strip().lower() for k in keywords]
                    initial_count = len(df_today)
                    
                    if 'text' in df_today.columns:
                        df_today = df_today[df_today['text'].astype(str).str.lower().apply(
                            lambda x: any(k in x for k in keyword_list)
                        )]
                    
                    if df_today.empty:
                        st.warning(f"⚠️ No articles found for {stock_filter}. Showing all stocks instead.")
                        df_today = load_today_articles(days_back=days_back)
                    else:
                        st.info(f"🔍 Filtered to {len(df_today)} articles about {stock_filter} (from {initial_count} total)")
                
                # Display articles
                st.markdown("---")
                st.subheader("📰 Article Preview")
                
                # Dynamically select available columns
                available_cols = []
                for col in ['title', 'text', 'source_name', 'source']:
                    if col in df_today.columns:
                        available_cols.append(col)
                
                # Fallback to just text if nothing else available
                if not available_cols:
                    available_cols = ['text'] if 'text' in df_today.columns else df_today.columns.tolist()[:3]
                
                display_df = df_today[available_cols].head(10)
                st.dataframe(display_df, use_container_width=True)
                
                # Process articles for sentiment
                st.markdown("---")
                st.subheader("🧠 Sentiment Analysis")
                
                with st.spinner("Analyzing sentiment for all articles..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    sentiments = []
                    scores = []
                    text_col = 'text'
                    
                    for i, (index, row) in enumerate(df_today.iterrows()):
                        text = row.get(text_col, '')
                        status_text.text(f"Analyzing article {i+1}/{len(df_today)}...")
                        progress_bar.progress((i + 1) / len(df_today))
                        
                        sent, _ = analyze_text(str(text)[:1000])  # Limit to 1000 chars for speed
                        score = get_sentiment_score(sent)
                        
                        sentiments.append(sent)
                        scores.append(score)
                    
                    df_today['sentiment'] = sentiments
                    df_today['sentiment_score'] = scores
                    
                    status_text.empty()
                    progress_bar.empty()
                
                # Display sentiment distribution
                col1, col2, col3 = st.columns(3)
                positive_count = scores.count(1)
                negative_count = scores.count(-1)
                neutral_count = scores.count(0)
                
                with col1:
                    st.metric("✅ Positive", f"{positive_count} ({positive_count/len(scores)*100:.1f}%)", 
                             delta="Bullish" if positive_count > len(scores)/3 else None)
                with col2:
                    st.metric("❌ Negative", f"{negative_count} ({negative_count/len(scores)*100:.1f}%)",
                             delta="Bearish" if negative_count > len(scores)/3 else None)
                with col3:
                    st.metric("➖ Neutral", f"{neutral_count} ({neutral_count/len(scores)*100:.1f}%)")
                
                # Generate Senior Analyst Summary
                st.markdown("---")
                st.subheader("📊 Senior Analyst Executive Summary")
                
                with st.spinner("🧠 Senior Analyst synthesizing market intelligence..."):
                    articles_list = df_today[text_col].astype(str).tolist()
                    report_name = stock_filter if stock_filter != "All Stocks" else "General Market"
                    
                    summary = generate_market_report(
                        articles_list, 
                        report_name, 
                        scores
                    )
                    
                    st.markdown(summary)
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="💾 Download Executive Summary",
                            data=summary,
                            file_name=f"Market_Digest_{datetime.now().strftime('%Y-%m-%d')}_{report_name.replace(' ', '_')}.txt",
                            mime="text/plain"
                        )
                    with col2:
                        csv_data = df_today[['text', 'sentiment', 'sentiment_score']].to_csv(index=False)
                        st.download_button(
                            label="📊 Download Sentiment Data (CSV)",
                            data=csv_data,
                            file_name=f"Sentiment_Data_{datetime.now().strftime('%Y-%m-%d')}.csv",
                            mime="text/csv"
                        )

# === TAB 2: LIVE ANALYSIS ===
with tab2:
    col1, col2 = st.columns([1.8, 1.2], gap="large")

    with col1:
        st.markdown("### 📰 News Analysis")
        news_text = st.text_area(
            "Input News Article", 
            height=180, 
            placeholder="Paste financial news here (e.g., 'CIB reports 30% profit growth in Q3...')...",
            label_visibility="collapsed"
        )
        
        analyze_btn = st.button("⚡ Analyze Sentiment", type="primary", use_container_width=True)

        if analyze_btn and news_text:
            with st.spinner("Analyzing financial aspects..."):
                # Call new ABSA engine
                from inference_engine import analyze_aspect_sentiment
                result = analyze_aspect_sentiment(news_text)
                
                if "error" in result:
                    st.error(f"Analysis Error: {result['error']}")
                else:
                    # --- General Sentiment Summary ---
                    gen_sent = result.get('general_sentiment', 'Neutral')
                    gen_reason = result.get('general_reasoning', 'No reasoning provided.')
                    
                    color = "#00CC96" if gen_sent.lower() == "positive" else "#EF553B" if gen_sent.lower() == "negative" else "#FFD700"
                    
                    st.markdown(f"""
                    <div style="padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.05); border-left: 5px solid {color}; margin-bottom: 20px;">
                        <h3 style="margin:0; color: {color}; text-transform: uppercase;">{gen_sent}</h3>
                        <p style="margin-top: 5px; font-size: 1em; color: #ccc;">{gen_reason}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # --- Visualization: Radar Chart ---
                    categories = ['Fundamentals', 'Macro/Reg', 'Market Sent.', 'Geopolitics']
                    scores = [
                        result['fundamentals']['score'],
                        result['macro']['score'],
                        result['sentiment']['score'],
                        result['geopolitics']['score']
                    ]
                    
                    # Close the loop for radar chart
                    categories.append(categories[0])
                    scores.append(scores[0])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=scores,
                        theta=categories,
                        fill='toself',
                        name='Impact',
                        line_color='#00CC96'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[-1, 1]
                            )
                        ),
                        showlegend=False,
                        height=350,
                        margin=dict(l=40, r=40, t=20, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- Graph Interpretation ---
                    graph_expl = result.get('graph_explanation', 'No interpretation available.')
                    st.caption(f"ℹ️ **Chart Interpretation**: {graph_expl}")
                    
                    # --- Detailed Reasoning ---
                    st.markdown("### 📝 Aspect Details")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.info(f"**Fundamentals**: {result['fundamentals']['score']}\n\n_{result['fundamentals']['reasoning']}_")
                        st.info(f"**Macro/Reg**: {result['macro']['score']}\n\n_{result['macro']['reasoning']}_")
                    with c2:
                        st.info(f"**Sentiment**: {result['sentiment']['score']}\n\n_{result['sentiment']['reasoning']}_")
                        st.info(f"**Geopolitics**: {result['geopolitics']['score']}\n\n_{result['geopolitics']['reasoning']}_")

    with col2:
        st.markdown(f"### 📊 {selected_name}")
        
        # Fetch Data
        try:
            stock = yf.Ticker(selected_ticker)
            hist = stock.history(period="3mo")
            
            if not hist.empty:
                # Calculate Metrics
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                change = current_price - prev_price
                pct_change = (change / prev_price) * 100
                
                # Color for price change
                delta_color = "normal" 
                
                st.metric(
                    label="Last Close (EGP)", 
                    value=f"{current_price:.2f}", 
                    delta=f"{change:.2f} ({pct_change:.2f}%)",
                    delta_color=delta_color
                )
                
                # Interactive Chart
                fig = go.Figure(data=[go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    increasing_line_color='#00CC96', 
                    decreasing_line_color='#EF553B'
                )])
                
                fig.update_layout(
                    height=350,
                    margin=dict(l=0, r=0, t=20, b=0),
                    xaxis_rangeslider_visible=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#888"),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#333')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # --- Statistical Validation (Granger Causality) ---
                st.markdown("---")
                st.markdown("### 📉 Statistical Validation")
                
                if st.button("🔍 Test Predictive Power", use_container_width=True):
                    with st.spinner("Running Granger Causality Test..."):
                        try:
                            # 1. Load Historical News
                            news_path = "/app/data/news/testing_data.jsonl"
                            if os.path.exists(news_path):
                                news_df = pd.read_json(news_path, lines=True)
                                
                                # Ensure 'date' column exists
                                if 'date' not in news_df.columns:
                                    if 'timestamp' in news_df.columns:
                                        news_df['date'] = pd.to_datetime(news_df['timestamp'], errors='coerce').dt.date
                                    elif 'published' in news_df.columns:
                                        news_df['date'] = pd.to_datetime(news_df['published'], errors='coerce').dt.date
                                    else:
                                        st.error("News data missing 'date' or 'timestamp' column.")
                                        news_df = pd.DataFrame()

                                if 'sentiment_score' not in news_df.columns:
                                    sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
                                    if 'sentiment' in news_df.columns:
                                        news_df['sentiment_score'] = news_df['sentiment'].map(sentiment_map)
                                    else:
                                        st.warning("News data missing sentiment scores.")
                                        news_df = pd.DataFrame()
                                
                                # Drop rows with missing dates or scores
                                news_df = news_df.dropna(subset=['date', 'sentiment_score'])
                                
                                # 2. Fetch Long-Term Stock Data
                                stock_hist = stock.history(period="1y")
                                stock_hist = stock_hist.reset_index()
                                
                                # 3. Run Test
                                if not news_df.empty and not stock_hist.empty:
                                    from analytics import calculate_granger_causality
                                    result = calculate_granger_causality(news_df, stock_hist, max_lag=5)
                                    
                                    if "error" in result:
                                        st.error(f"Test Failed: {result['error']}")
                                    else:
                                        p_val = result['min_p_value']
                                        is_sig = result['is_significant']
                                        lag = result['best_lag']
                                        
                                        if is_sig:
                                            st.success(f"✅ **Significant Predictive Power** (p={p_val:.4f})")
                                            st.caption(f"News sentiment significantly predicts price movements {lag} days later.")
                                        else:
                                            st.warning(f"❌ **No Significant Causality** (p={p_val:.4f})")
                                            st.caption("Current news sentiment does not statistically predict price movements for this stock.")
                            else:
                                st.error("Historical news data not found.")
                        except Exception as e:
                            st.error(f"Validation Error: {e}")
                
                # Volume Bar
                st.caption("Volume (3mo)")
                st.bar_chart(hist['Volume'], height=100, color="#333333")
            else:
                st.warning(f"⚠️ Market data unavailable for {selected_ticker}")
                st.caption(f"Could not fetch data for {selected_name}. This might be due to a temporary API issue or the stock being delisted.")
                
                # Fallback: Show empty chart or placeholder
                st.info("💡 You can still use the News Analysis feature on the left.")

        except Exception as e:
            st.error(f"⚠️ Market Data Error: {str(e)}")
            st.caption("Try selecting a different company or check your internet connection.")
            st.info("💡 You can still use the News Analysis feature on the left.")
            # logging.error(f"Stock fetch error: {e}")
            
# === TAB 3: BATCH PROCESSING ===
with tab3:
    st.markdown("### 🏭 Feature Extraction for Forecasting")
    st.markdown("Upload your historical news data and select a target stock. The app will automatically filter for relevant articles (using English/Arabic keywords) and generate sentiment scores.")
    
    uploaded_file = st.file_uploader("Upload CSV or JSONL", type=["csv", "jsonl"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file, lines=True)
            
            st.dataframe(df.head(), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                text_col = st.selectbox("Select Text Column", df.columns)
            with col2:
                date_col = st.selectbox("Select Date Column (Optional)", ["None"] + list(df.columns))
            with col3:
                # Smart Filter Dropdown
                target_stock = st.selectbox("Select Target Stock", ["None (Process All)"] + list(STOCK_DATA.keys()))
            
            # Filter Logic
            if target_stock != "None (Process All)":
                keywords = STOCK_DATA[target_stock]["keywords"]
                st.info(f"🔍 Filtering for **{target_stock}** using keywords: {', '.join(keywords)}")
                
                keyword_list = [k.strip().lower() for k in keywords]
                initial_count = len(df)
                # Filter rows where text contains ANY of the keywords
                df = df[df[text_col].astype(str).str.lower().apply(lambda x: any(k in x for k in keyword_list))]
                final_count = len(df)
                
                if final_count == 0:
                    st.warning("⚠️ No articles matched the selected stock. Try 'None' to process all.")
                else:
                    st.success(f"✅ Found **{final_count}** relevant articles (out of {initial_count}).")
            
            # Aggregation Option
            aggregate_daily = False
            if date_col != "None":
                aggregate_daily = st.checkbox("📅 Aggregate Scores by Day? (Recommended for Forecasting)", value=True)

            if st.button("🚀 Start Batch Processing", disabled=df.empty):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                sentiments = []
                scores = []
                
                total = len(df)
                start_time = time.time()
                
                # Iterate over the filtered dataframe
                for i, (index, row) in enumerate(df.iterrows()):
                    text = row[text_col]
                    
                    # Update UI
                    status_text.text(f"Processing {i+1}/{total}: {str(text)[:50]}...")
                    progress_bar.progress((i + 1) / total)
                    
                    # Inference
                    sent, _ = analyze_text(str(text))
                    score = get_sentiment_score(sent)
                    
                    sentiments.append(sent)
                    scores.append(score)
                
                # Add results
                df['sentiment'] = sentiments
                df['sentiment_score'] = scores
                
                # Handle Aggregation
                if aggregate_daily and date_col != "None":
                    try:
                        # Convert to datetime
                        df[date_col] = pd.to_datetime(df[date_col])
                        # Group by Date
                        daily_df = df.groupby(df[date_col].dt.date).agg({
                            'sentiment_score': 'mean',
                            text_col: 'count'  # Count articles per day
                        }).reset_index()
                        daily_df.rename(columns={text_col: 'article_count', 'sentiment_score': 'daily_sentiment_score'}, inplace=True)
                        
                        st.success(f"✅ Aggregated into {len(daily_df)} daily records!")
                        st.dataframe(daily_df.head(), use_container_width=True)
                        
                        # Download Aggregated
                        filename = f"{target_stock.replace(' ', '_')}_DAILY_features.csv" if target_stock != "None (Process All)" else "daily_sentiment_features.csv"
                        csv = daily_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"💾 Download Daily Features CSV",
                            data=csv,
                            file_name=filename,
                            mime='text/csv',
                        )
                    except Exception as e:
                        st.error(f"Aggregation Failed: {e}")
                        # Fallback to raw download
                        st.warning("Downloading raw data instead.")
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="💾 Download Raw Features CSV",
                            data=csv,
                            file_name='raw_features.csv',
                            mime='text/csv',
                        )
                else:
                    end_time = time.time()
                    duration = end_time - start_time
                    st.success(f"✅ Processed {total} items in {duration:.2f} seconds!")
                    
                    # Preview
                    st.dataframe(df[[text_col, 'sentiment', 'sentiment_score']].head(), use_container_width=True)
                    
                    # Download Raw
                    filename = f"{target_stock.replace(' ', '_')}_features.csv" if target_stock != "None (Process All)" else "egysentiment_features.csv"
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"💾 Download {filename}",
                        data=csv,
                        file_name=filename,
                        mime='text/csv',
                    )
            
            # --- Generative Report Section ---
            st.markdown("---")
            st.subheader("📝 Generative Market Report (AI Senior Analyst)")
            st.caption("Generate a professional executive summary based on the filtered articles. Our AI analyst will synthesize market sentiment, identify key drivers, and provide actionable insights.")
            
            if st.button("✨ Generate Executive Summary", disabled=df.empty):
                with st.spinner("🧠 Senior Analyst is synthesizing market data..."):
                    # Get list of texts
                    articles_list = df[text_col].astype(str).tolist()
                    report_name = target_stock if target_stock != "None (Process All)" else "General Market"
                    
                    # Pass sentiment scores if they exist in the dataframe
                    sentiment_scores_list = None
                    if 'sentiment_score' in df.columns:
                        sentiment_scores_list = df['sentiment_score'].tolist()
                    
                    summary = generate_market_report(articles_list, report_name, sentiment_scores_list)
                    
                    st.success("✅ Executive Summary Generated Successfully!")
                    st.markdown(f"### 📊 Executive Summary: {report_name}")
                    st.markdown(summary)
                    
                    # Download Report
                    st.download_button(
                        label="💾 Download Report (TXT)",
                        data=summary,
                        file_name=f"{report_name.replace(' ', '_')}_Executive_Summary.txt",
                        mime="text/plain"
                    )
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>EgySentiment © 2024 | Financial Intelligence Unit</div>", unsafe_allow_html=True)

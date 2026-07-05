import os
import json
from urllib import response
import pandas as pd
import yfinance as yf
import ta
import numpy as np
import ollama
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime, timedelta, timezone
import difflib
import time
from typing import List, Dict, Any
import urllib
import re
from collections import defaultdict
import feedparser
from bs4 import BeautifulSoup
import requests
# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb

from langchain_ollama import OllamaEmbeddings
from gnews import GNews

# Project Config Imports
from config import TICKER, RAG_MODEL, SENTIMENT_MODEL, SYNTHESIS_MODEL, EMBEDDING_MODEL, GEMINI_API_KEY,ARTICLE_LOOKBACK, SIMILARITY_THRESHOLD, MAX_PER_DOMAIN, MAX_RETRIES

# ==========================================
# STEP 0: Pure Free SEC Disclosures Engine
# ==========================================
def fetch_sec_financial_text(ticker: str, api_key: str = None) -> str:
    """
    Fetches future guidance and analyst Q&A from the latest earnings call 
    for a given ticker using LangChain and Gemini with Search Grounding 
    and a self-reflection prompt.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').
        api_key (str, optional): Your Gemini API key. If None, it defaults 
                                 to the GEMINI_API_KEY environment variable.
                                 
    Returns:
        str: The full self-reflected output from the model.
    """
    # Resolve the API key
    resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY")
    
    # Initialize the LangChain chat model wrapper for Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=resolved_api_key,
        # Slight temperature control to keep financial numbers stable and factual
        temperature=0.2 
    )
    
    # Constructing the prompt with an explicit self-reflection loop
    prompt = f"""
    You are an expert Wall Street financial analyst. Your task is to find the transcript or 
    comprehensive summary of the **latest quarterly earnings call** for the stock ticker: {ticker}.
    
    Execute your task using the following strict 3-step Self-Reflection process. You must output 
    all three steps so I can verify your reasoning.

    ---
    ### Step 1: First Draft Extraction
    Search for the latest earnings call data. Identify and extract:
    1. Management's future guidance (revenue targets, EPS expectations, margins, or qualitative macroeconomic outlooks).
    2. The most critical analyst Q&A pairs (the core questions asked by analysts and the exact substance of management's answers).

    ---
    ### Step 2: Self-Reflection & Critique
    Critically review your first draft. Evaluate it based on the following:
    - Did I verify that this is truly from the *most recent* earnings call, or am I pulling from a previous quarter?
    - Are the specific financial numbers and metrics cited completely accurate according to the sources?
    - Did I leave out vital context from the Q&A that changes the meaning of management's answers?
    - Is there any boilerplate language or fluff that should be condensed for maximum impact?

    ---
    ### Step 3: Final Polished Output
    Based on your critique in Step 2, rewrite and refine the findings into a flawless, publication-grade executive summary. 
    Organize this final section neatly with distinct headers for:
    - **Executive Summary & Date of Call**
    - **Refined Future Guidance & Financial Outlook**
    - **Key Analyst Q&A Pairs (Contextualized)**
    """

    try:
        # CRITICAL: Bind the Google Search tool natively using LangChain's syntax 
        # to fetch up-to-date live market web transcripts
        llm_with_search = llm.bind_tools([{"google_search": {}}])
        
        # Invoke the chain
        response = llm_with_search.invoke(prompt)
        
        # LangChain stores the text output in the .content attribute
        return response.content
        
    except Exception as e:
        return f"An error occurred while generating content: {str(e)}"
    
    
# ==========================================
# STEP 1: Technical Analysis (Pure Python)
# ==========================================
def run_step_1_technical(ticker_symbol: str) -> str:
    """
    Fetches historical data, calculates indicators, and returns a formatted 
    string of ONLY the latest values.
    """
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="2y")

    if df.empty:
        return f"Error: No data available for {ticker_symbol}."

    def rma(series, period):
        return series.ewm(alpha=1/period, adjust=False).mean()

    # 1. Price & Volume
    df['ClosePrice'] = df['Close']
    
    # 2. RSI14
    delta = df['ClosePrice'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    rs = rma(gain, 14) / rma(loss, 14)
    df['RSI14'] = 100 - (100 / (1 + rs))

    # 3. MACD
    ema12 = df['ClosePrice'].ewm(span=12, adjust=False).mean()
    ema26 = df['ClosePrice'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACDSignal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACDHistogram'] = df['MACD'] - df['MACDSignal']

    # 4. Moving Averages
    df['SMA50'] = df['ClosePrice'].rolling(window=50).mean()
    df['SMA200'] = df['ClosePrice'].rolling(window=200).mean()
    df['EMA20'] = df['ClosePrice'].ewm(span=20, adjust=False).mean()

    # 5. Volume
    df['AverageVolume20'] = df['Volume'].rolling(window=20).mean()
    df['RelativeVolume'] = df['Volume'] / df['AverageVolume20']

    # 6. ATR14
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['ClosePrice'].shift()).abs()
    low_close = (df['Low'] - df['ClosePrice'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = rma(tr, 14)

    # 7. ADX14
    up_move = df['High'] - df['High'].shift()
    down_move = df['Low'].shift() - df['Low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * (rma(pd.Series(plus_dm, index=df.index), 14) / df['ATR14'])
    minus_di = 100 * (rma(pd.Series(minus_dm, index=df.index), 14) / df['ATR14'])
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    df['ADX14'] = rma(dx, 14)

    # 8. Bollinger Band Width
    sma20 = df['ClosePrice'].rolling(window=20).mean()
    std20 = df['ClosePrice'].rolling(window=20).std()
    upper_band = sma20 + (2 * std20)
    lower_band = sma20 - (2 * std20)
    df['BollingerBandWidth'] = (upper_band - lower_band) / sma20

    # 9. OBV
    direction = np.sign(df['ClosePrice'].diff()).fillna(1)
    df['OBV'] = (df['Volume'] * direction).cumsum()

    # 10. Distance from MAs
    df['DistanceFromSMA50Pct'] = ((df['ClosePrice'] - df['SMA50']) / df['SMA50']) * 100
    df['DistanceFromSMA200Pct'] = ((df['ClosePrice'] - df['SMA200']) / df['SMA200']) * 100

    # 11. Returns
    df['DailyReturnPct'] = df['ClosePrice'].pct_change() * 100
    df['Return5DPct'] = df['ClosePrice'].pct_change(periods=5) * 100
    df['Return20DPct'] = df['ClosePrice'].pct_change(periods=20) * 100

    # 12. 52-Week High/Low (252 trading days)
    rolling_252_high = df['High'].rolling(window=252).max()
    rolling_252_low = df['Low'].rolling(window=252).min()
    df['DistanceFrom52WeekHighPct'] = ((df['ClosePrice'] - rolling_252_high) / rolling_252_high) * 100
    df['DistanceFrom52WeekLowPct'] = ((df['ClosePrice'] - rolling_252_low) / rolling_252_low) * 100

    # 13. MA Breakouts
    df['AboveSMA50'] = df['ClosePrice'] > df['SMA50']
    df['AboveSMA200'] = df['ClosePrice'] > df['SMA200']

    # 14. Golden Cross
    df['GoldenCross'] = (df['SMA50'] > df['SMA200']) & (df['SMA50'].shift(1) <= df['SMA200'].shift(1))

    columns_to_return = [
        'ClosePrice', 'RSI14', 'MACD', 'MACDSignal', 'MACDHistogram',
        'SMA50', 'SMA200', 'EMA20', 'Volume', 'AverageVolume20',
        'RelativeVolume', 'ATR14', 'ADX14', 'BollingerBandWidth', 'OBV',
        'DistanceFromSMA50Pct', 'DistanceFromSMA200Pct', 'DailyReturnPct',
        'Return5DPct', 'Return20DPct', 'DistanceFrom52WeekHighPct',
        'DistanceFrom52WeekLowPct', 'AboveSMA50', 'AboveSMA200', 'GoldenCross'
    ]

    # Drop NA to ensure we have a valid final row, then isolate the very last day
    valid_data = df[columns_to_return].dropna()
    if valid_data.empty:
        return f"Error: Not enough historical data to calculate all indicators for {ticker_symbol}."
        
    latest_data = valid_data.iloc[-1]

    # Build the string output
    output_lines = [f"--- Latest Technical Indicators for {ticker_symbol} ---"]
    for indicator, value in latest_data.items():
        if isinstance(value, float):
            # Format floats to 4 decimal places
            output_lines.append(f"{indicator}: {value:.4f}")
        else:
            # Handle booleans and integers
            output_lines.append(f"{indicator}: {value}")

    return "\n".join(output_lines)

# ==========================================
# STEP 2: Corporate Filing Local RAG Engine
# ==========================================
def run_step_2_rag(transcript_text: str) -> str:
    print(f"\n--- Step 2: Extracting Insights via Native ChromaDB ({RAG_MODEL}) ---")
    
    # 1. Chunk the plain text document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.create_documents([transcript_text])
    
    # Extract structural plain string elements from chunk objects
    chunks = [doc.page_content for doc in split_docs]
    ids = [f"chunk_id_{idx}" for idx in range(len(chunks))]
    
    # 2. Build local embeddings explicitly using Ollama to store in Chroma
    print(f"[Embedding] Processing {len(chunks)} text chunks via '{EMBEDDING_MODEL}'...")
    embeddings_list = []
    for chunk in chunks:
        embed_resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=chunk)
        embeddings_list.append(embed_resp['embedding'])
    
    # 3. Instantiate an ephemeral in-memory Chroma Client (no resource disk locks)
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.create_collection(name="financial_analysis")
    
    # Feed native vectors and documents straight into the collection
    collection.add(
        ids=ids,
        embeddings=embeddings_list,
        documents=chunks
    )
    
    queries = [
        "What is management's forward margin guidance and revenue outlook?",
        "What are the primary operational risks or macro headwinds mentioned?",
        "What critical questions or concerns did analysts raise during the Q&A?"
    ]
    
    summaries = []
    
    # 4. Sequentially process queries
    for query in queries:
        # Generate the query embedding using the same model
        query_embed_resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
        query_vector = query_embed_resp['embedding']
        
        # Query ChromaDB directly using vectors
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=2
        )
        
        # Pull text components out of structural nested dictionary response
        retrieved_texts = results['documents'][0] if results['documents'] else []
        context = "\n\n".join(retrieved_texts)
        
        # Generation handoff using llama3.1:8b
        response = ollama.generate(
            model=RAG_MODEL,
            system="Answer using bullets based strictly on the context. If absent, state 'Not discussed'.",
            prompt=f"Context:\n{context}\n\nQuery: {query}",
            options={"num_ctx": 16384, "temperature": 0.0}
        )
        summaries.append(f"### {query}\n{response['response'].strip()}\n")
        
    return "\n".join(summaries)

# ==========================================
# STEP 3: News Aggregator & Sentiment Classification
# ==========================================

def get_company_name(ticker: str) -> str:
    """
    Fetches the company name for a given ticker using yfinance.
    Cleans up common corporate suffixes for better search results.
    """
    try:
        stock = yf.Ticker(ticker)
        # shortName usually contains the clean name (e.g., "Apple Inc.")
        name = stock.info.get('displayName', ticker)
    
        # Remove common suffixes to improve search accuracy 
        # (e.g., "Apple Inc." -> "Apple")
        clean_name = re.sub(r'(?i)\b(inc|corp|corporation|ltd|plc|llc)\b\.?', '', name).strip()
        # Remove trailing commas or hyphens
        clean_name = re.sub(r'[,/-]$', '', clean_name).strip()
        
        return clean_name if clean_name else ticker
    except Exception:
        # Fallback to the ticker itself if lookup fails
        return ticker

def run_step_3_sentiment(
    ticker: str,
    days_back: int = 3, 
    similarity_threshold: float = 0.85,
    max_per_domain: int = 3,
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    """
    Fetches investment news using a hybrid resilient architecture:
    1. Heuristic Filter: Caps max articles per source domain.
    2. Domain Interleaving: Prevents back-to-back requests to the same server.
    3. Exponential Backoff: Handles 429 Too Many Requests errors gracefully.
    """
    ticker = ticker.upper()
    company_name = get_company_name(ticker)
    
    encoded_name = urllib.parse.quote(f'"{company_name}"')
    rss_feeds = [
        f"https://finance.yahoo.com/rss/headline?s={ticker}",
        f"https://news.google.com/rss/search?q={ticker}+OR+{encoded_name}+when:{days_back}d"
    ]
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    raw_articles = []
    
    # --- Phase 1: Fetch and Deduplicate Metadata ---
    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                pub_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    pub_date = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                
                if pub_date and pub_date >= cutoff_date:
                    raw_articles.append({
                        "title": getattr(entry, 'title', '').strip(),
                        "link": getattr(entry, 'link', '').strip(),
                        "published": pub_date.isoformat(),
                        "source_domain": urllib.parse.urlparse(feed_url).netloc,
                        "ticker": ticker,
                        "company_name": company_name
                    })
        except Exception:
            continue

    # Deduplicate by title similarity
    unique_articles = []
    for article in raw_articles:
        if not article["title"] or not article["link"]:
            continue
            
        is_duplicate = False
        for unique in unique_articles:
            if article["link"] == unique["link"]:
                is_duplicate = True
                break
            
            similarity = difflib.SequenceMatcher(None, article["title"].lower(), unique["title"].lower()).ratio()
            if similarity >= SIMILARITY_THRESHOLD:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_articles.append(article)
            
    # Sort chronologically so we prioritize the newest articles when capping
    unique_articles.sort(key=lambda x: x['published'], reverse=True)

    # --- Phase 2: Domain Grouping & Capping (Design 1) ---
    domain_groups = defaultdict(list)
    for article in unique_articles:
        domain = article["source_domain"]
        if len(domain_groups[domain]) < max_per_domain:
            domain_groups[domain].append(article)

    # --- Phase 3: Queue Interleaving (Design 3) ---
    # Create a round-robin queue to space out domain requests
    interleaved_queue = []
    while any(domain_groups.values()):
        for domain in list(domain_groups.keys()):
            if domain_groups[domain]:
                interleaved_queue.append(domain_groups[domain].pop(0))

    # --- Phase 4: Execution with Exponential Backoff (Design 3) ---
    investment_events = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for article in interleaved_queue:
        retries = 0
        backoff_delay = 5  # Start with a 5-second timeout on a 429
        
        while retries <= max_retries:
            try:
                time.sleep(1) # Baseline polite delay between all requests
                
                response = requests.get(article["link"], headers=headers, timeout=10, allow_redirects=True)
                
                # Handle Too Many Requests
                if response.status_code == 429:
                    retries += 1
                    if retries <= max_retries:
                        print(f"HTTP 429 encountered on {article['source_domain']}. Backing off for {backoff_delay}s...")
                        time.sleep(backoff_delay)
                        backoff_delay *= 2  # Exponentially increase delay
                    continue
                
                # Process Successful Request
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button"]):
                        element.extract()
                    
                    paragraphs = soup.find_all('p')
                    full_text = " ".join([p.get_text().strip() for p in paragraphs if p.get_text()])
                    cleaned_text = " ".join(full_text.split())
                    
                    if len(cleaned_text) > 200: 
                        investment_events.append({
                            "event_id": hash(article["link"]),
                            "ticker": article["ticker"],
                            "company_name": article["company_name"],
                            "title": article["title"],
                            "timestamp": article["published"],
                            "original_url": response.url,
                            "full_text": cleaned_text
                        })
                    break # Break out of the retry loop on success
                else:
                    break # Break on 404, 500, etc. (No need to retry non-rate-limit errors)
                    
            except requests.exceptions.RequestException:
                break # Move to the next article if connection drops completely

    # Final sort to return to the agent chronologically
    investment_events.sort(key=lambda x: x['timestamp'], reverse=True)
    return investment_events

# ==========================================
# STEP 5: Intrinsic value metrics extraction from raw financial statements
# ==========================================

def extract_intrinsic_value_metrics(ticker_symbol: str) -> Dict[str, Any]:
    """
    Extracts multi-year fundamental, capital allocation, valuation, and moat metrics
    including R&D, CapEx, EV/EBITDA, and Historical Percentiles.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # 1. Fetch raw financial dataframes
        financials = ticker.financials      # Income Statement
        balance = ticker.balance_sheet      # Balance Sheet
        cashflow = ticker.cashflow          # Cash Flow Statement
        info = ticker.info                  # Valuation multiples
        
        if financials.empty or balance.empty or cashflow.empty:
            raise ValueError("Financial statements could not be retrieved.")

        df_fin = financials.T
        df_bal = balance.T
        df_cash = cashflow.T
        
        common_years = sorted(df_fin.index.intersection(df_bal.index).intersection(df_cash.index))
        historical_metrics = []
        
        for year in common_years:
            try:
                # Core fundamental extractions
                ebit = df_fin.loc[year, 'EBIT'] if 'EBIT' in df_fin.columns else 0
                tax_expense = df_fin.loc[year, 'Tax Provision'] if 'Tax Provision' in df_fin.columns else 0
                total_rev = df_fin.loc[year, 'Total Revenue'] if 'Total Revenue' in df_fin.columns else 1 # Avoid div by zero
                
                # New Moat/Capital Intensity Metrics
                rnd = df_fin.loc[year, 'Research And Development'] if 'Research And Development' in df_fin.columns else 0
                capex = df_cash.loc[year, 'Capital Expenditure'] if 'Capital Expenditure' in df_cash.columns else 0
                
                rnd_pct = (rnd / total_rev) * 100
                capex_sales_ratio = (abs(capex) / total_rev) * 100
                gross_profit = df_fin.loc[year, 'Gross Profit']
                
                # ROIC Calculation
                effective_tax_rate = max(0.0, min(tax_expense / ebit, 0.5)) if ebit > 0 else 0.21
                nopat = ebit * (1 - effective_tax_rate)
                
                total_debt = (df_bal.loc[year, 'Total Debt'] if 'Total Debt' in df_bal.columns else 0)
                total_equity = df_bal.loc[year, 'Stockholders Equity'] if 'Stockholders Equity' in df_bal.columns else 0
                cash_equiv = df_bal.loc[year, 'Cash And Cash Equivalents'] if 'Cash And Cash Equivalents' in df_bal.columns else 0
                
                invested_capital = total_debt + total_equity - cash_equiv
                roic = (nopat / invested_capital) * 100 if invested_capital > 0 else 0
                
                 # Margins
                gross_margin = (gross_profit / total_rev) * 100
                operating_margin = (ebit / total_rev) * 100
                
                # --- PILLAR 4: CAPITAL ALLOCATION ---
                free_cash_flow = df_cash.loc[year, 'Free Cash Flow'] if 'Free Cash Flow' in df_cash.columns else 0
                shares_outstanding = df_bal.loc[year, 'Share Capital'] if 'Share Capital' in df_bal.columns else None
                # Alternative if Share Capital is not structured: use info for latest, track trend via common stock values
                common_stock = df_bal.loc[year, 'Common Stock'] if 'Common Stock' in df_bal.columns else 0



                historical_metrics.append({
                    "Year": str(year.date()),
                    "ROIC (%)": round(roic, 2),
                    "Gross Margin (%)": round(gross_margin, 2),
                    "Operating Margin (%)": round(operating_margin, 2),
                    "Free Cash Flow ($)": free_cash_flow,
                    "Total Debt to Equity": round(total_debt / total_equity, 2) if total_equity > 0 else 0,
                    "R&D as % of Revenue": round(rnd_pct, 2),
                    "CapEx to Sales (%)": round(capex_sales_ratio, 2),
                    "Debt to Equity": round(total_debt / total_equity, 2) if total_equity > 0 else 0
                })
            except KeyError:
                continue

        # 2. Current Valuation & Percentiles
        current_price = info.get("currentPrice", 0)
        
        # Calculate 5-Year Historical Valuation Percentile
        hist_data = ticker.history(period="5y", interval="1wk")
        valuation_percentile = "N/A"
        if not hist_data.empty and current_price > 0:
            five_yr_low = hist_data['Close'].min()
            five_yr_high = hist_data['Close'].max()
            if five_yr_high > five_yr_low:
                pct_val = ((current_price - five_yr_low) / (five_yr_high - five_yr_low)) * 100
                valuation_percentile = f"{round(pct_val, 1)}% (0% = 5yr Low, 100% = 5yr High)"

        valuation_metrics = {
            "Enterprise Value to EBITDA (EV/EBITDA)": info.get("enterpriseToEbitda", "N/A"),
            "5-Year Valuation Range Percentile": valuation_percentile,
            "Price to Free Cash Flow (P/FCF)": round(info.get("marketCap", 0) / info.get("freeCashflow", 1), 2) if info.get("freeCashflow", 0) > 0 else "N/A",
            "Forward P/E": info.get("forwardPE", "N/A"),
            "PEG Ratio": info.get("pegRatio", "N/A")
        }

        return {
            "status": "SUCCESS",
            "ticker": ticker_symbol,
            "historical_trends": historical_metrics,
            "current_valuation": valuation_metrics
        }

    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def format_for_llm(metrics_dict: Dict[str, Any]) -> str:
    """Formats the metrics cleanly for the DeepSeek-R1 prompt."""
    if metrics_dict["status"] == "ERROR":
        return f"Error: {metrics_dict['message']}"
        
    out = f"--- FINANCIAL METRICS FOR {metrics_dict['ticker']} ---\n\n"
    out += "Historical Moat & Capital Allocation:\n"
    for p in metrics_dict["historical_trends"]:
        out += f"- {p['Year']} | ROIC: {p['ROIC (%)']}% | Gross Margin: {p['Gross Margin (%)']}% | Operating Margin: {p['Operating Margin (%)']}% | Debt/Equity: {p['Total Debt to Equity']}% | R&D to Rev: {p['R&D as % of Revenue']}% | CapEx to Sales: {p['CapEx to Sales (%)']}%\n"
        
    out += "\nCurrent Valuation & Margin of Safety:\n"
    for metric, val in metrics_dict["current_valuation"].items():
        out += f"- {metric}: {val}\n"
    return out
# ==========================================
# STEP 4: Executive Committee Synthesis
# ==========================================
def run_step_4_synthesis(
    ticker: str, 
    earnings_snippet: str, 
    technical_indicators: str, 
    news_snippet: str
) -> Dict[str, Any]:
    """
    Leverages a local deepseek-r1-8b model via Ollama to synthesize fundamental,
    technical, and sentiment data into a structured investment action.
    """
    
    # 1. Format the technical indicators into a readable string for the LLM
    #tech_indicators_str = "\n".join([f"- {key}: {value}" for key, value in technical_indicators.items()])
    
    # 2. Define the optimized system and user prompt
    prompt = f"""You are a conservative, long-term intrinsic value investor adhering strictly to the principles of fundamental business analysis and margin of safety. Your objective is to evaluate the equity instrument {ticker} for a multi-year holding period.

### INPUT DATA ###

1. CORE FUNDAMENTALS & FUTURE GUIDANCE (Earnings Call Transcript):
{earnings_snippet}

2. MARKET SENTIMENT, COMPETITIVE MOAT & MACRO RISKS (News Snippets):
{news_snippet}

3. SHORT-TERM PRICE MOMENTUM (Technical Indicators):
{technical_indicators}

4. FINANCIAL HEALTH, MOAT & VALUATION METRICS:
{formatted_text}

### EVALUATION CRITERIA & COGNITIVE FRAMEWORK ###
Analyze the inputs using the following long-term investing rubric:

1. Business Trajectory & Capital Allocation (70% Weight):
  a. Evaluate management's forward guidance from the earnings call.
  b. Cross-reference their claims with the Financial Metrics (ROIC, CapEx to Sales, R&D spend). Are they maintaining their competitive moat? 
2. Structural & Macro Risks (15% Weight): Identify regulatory threats, supply chain dependencies, or industry shifts from the news.
3. Margin of Safety & Valuation (10% Weight): Use EV/EBITDA, P/FCF, and the 5-Year Valuation Percentile to determine if the stock is priced at a premium, fair value, or deep discount. Do not overpay for a good business. 
4. Technical Timing (5% Weight): Do not use technicals to change the underlying business thesis. Use them strictly to judge if the stock is currently in a hyper-extended momentum bubble (Overbought) or if short-term panic has created an accumulation window (Oversold).

### OUTPUT FORMAT INSTRUCTIONS ###
Provide your final evaluation strictly as a raw JSON object matching the keys below. Do not wrap it in markdown code blocks.

Template:
{{
"ticker": "{ticker}",
"allocation_decision": "ACCUMULATE" | "HOLD" | "AVOID",
"intrinsic_value_thesis": "Summary of the long-term compounding viability based on guidance and ROIC.",
"moat_and_risk_analysis": "Key structural risks or competitive advantages identified from R&D/CapEx and news.",
"margin_of_safety_assessment": "Assessment of current valuation multiples (EV/EBITDA, percentiles)",
"entry_timing_note": "Assessment of whether current technicals present a premium or discount entry point.",
"confidence_score": 0.85
}}"""

    try:
        # 3. Call the local Ollama instance
        # Using format='json' forces the model output to be structurally valid JSON
        response = ollama.chat(
            model=SYNTHESIS_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.2,  # Low temperature for highly analytical, deterministic output
                'top_p': 0.9,
                'num_ctx': 10192
            },
            format='json' 
        )
        
        # 4. Extract and parse the response content
        response_content = response['message']['content']
        decision_data = json.loads(response_content)
        return decision_data

    except json.JSONDecodeError as je:
        print(f"Error parsing JSON from model response: {je}")
        return {
            "ticker": ticker,
            "decision": "ERROR",
            "error_message": "Failed to parse model response as JSON",
            "raw_output": response.get('message', {}).get('content', '')
        }
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {
            "ticker": ticker,
            "decision": "ERROR",
            "error_message": str(e)
        }

# ==========================================
# Orchestration Pipeline
# ==========================================
if __name__ == "__main__":
    print(f"[Pipeline] Executing target pipeline sequence for: {TICKER}")
    
    # Process financial context completely for free via direct SEC archive extraction
    raw_financial_text = fetch_sec_financial_text(TICKER,GEMINI_API_KEY)
    #print(raw_financial_text)
    tech_data = run_step_1_technical(TICKER)
    #rag_data = run_step_2_rag(raw_financial_text)
    #print(tech_data)
    sentiment = run_step_3_sentiment(TICKER,ARTICLE_LOOKBACK, SIMILARITY_THRESHOLD, MAX_PER_DOMAIN, MAX_RETRIES)
    #print(sentiment)
    raw_data = extract_intrinsic_value_metrics(TICKER)
    formatted_text = format_for_llm(raw_data)
    #print(formatted_text)
    decision = run_step_4_synthesis(TICKER, raw_financial_text, tech_data, sentiment)
    print(decision)
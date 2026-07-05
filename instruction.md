# Specification: Local Financial Multi-Agent System (Sequential Pipeline)

## Target Architecture
Build a Python-based investment research pipeline that runs entirely on a single local desktop (16GB RAM). The pipeline must execute sequentially (step-by-step) to minimize memory usage, ensuring only one local model is actively loaded or processing heavily in Ollama at any given time.

## Tech Stack & Models (Updated)
- Framework: Python 3.11+, LangChain (or simple sequential functions using the 'ollama' Python SDK).
- Model 1 (Lightweight / Structuring / JSON): `qwen3:4b` (used for sentiment evaluation and strict JSON array data filtering).
- Model 2 (Reasoning / Dense Analysis): `deepseek-r1:8b` (used for deep con-call transcript analysis via chain-of-thought reasoning and final action recommendations).
- Vector DB: FAISS or Chroma (using local lightweight embeddings: `nomic-embed-text`).
---

## Sequential Workflow Steps

### Step 1: Technical & Numerical Pre-Processing (Pure Python)
- Do NOT use an LLM for this step.
- Implement a script using `pandas` and `ta` (or `TA-Lib`) to fetch daily price history for a user-specified stock ticker.
- Calculate: RSI (14), MACD (and signal line), 50-day SMA, and 200-day SMA.
- Output: A cleanly formatted Markdown table containing only the latest day's calculated technical values (not raw historical rows).

### Step 2: Con-Call Transcript Local RAG Pipeline (Model: llama3.1:8b)
- Load a local earnings call transcript text file or PDF.
- Use a text splitter to chunk the document into 1,000-token blocks with 200-token overlaps.
- Store chunks in the local vector database using local embeddings.
- Execute exactly 3 specific, targeted queries against the vector store:
    1. "What is management's forward margin guidance and revenue outlook?"
    2. "What are the primary operational risks or macro headwinds mentioned?"
    3. "What critical questions or concerns did analysts raise during the Q&A?"
- Pass the retrieved context chunks for these queries to `llama3.1:8b` (explicitly set context option `num_ctx: 16384` in the API call) to generate a concise bulleted summary for each query.
- Output: Save this text as `transcript_summary.txt`.

### Step 3: News Aggregator & Sentiment Classification (Model: llama3.2:3b)
- Use a library like `GNews` to fetch the top 5–7 news headlines and snippets for the selected ticker from the past 7 days.
- Format these items into a clean text block.
- Invoke `llama3.2:3b`. Use a strict System Prompt forcing it to output only a valid JSON array of numerical sentiment scores ranging from -1.0 (Highly Bearish) to +1.0 (Highly Bullish) for each headline.
- Output: Calculate the mathematical average of these scores in Python. Save the result as a variable (`news_sentiment_score`).

### Step 4: The Investment Committee Synthesis (Model: llama3.1:8b)
- Aggregate the structured data from the previous steps into a final comprehensive Markdown prompt.
- The prompt context must look like this:
  ### DATA INPUT
  - **Ticker**: [Ticker Name]
  - **Technical Profile**: [Markdown Table from Step 1]
  - **News Sentiment Score**: [Average Score from Step 3]
  - **Earnings Call Summary**: [Content from transcript_summary.txt]
- Invoke `llama3.1:8b` (set `num_ctx: 8192`) with a system prompt casting it as a Senior Investment Analyst.
- Instruct the model to review the compiled indicators and write a final concise summary containing:
    1. An Action Recommendation: Clear tag of **BUY**, **SELL**, or **HOLD**.
    2. Rationale: Exactly 3 high-impact bullet points justifying the call based on the data provided.

---

## Memory & Execution Constraints
1. **No Concurrency:** Do not use `asyncio.gather`, threading, or parallel tool calls. Ensure Step N completely finishes and drops its data buffers before Step N+1 initializes its LLM call.
2. **Ollama Automatic Purging:** Rely on Ollama's native behavior to offload the previous model when the next model size is requested. Ensure the code handles brief latency handoffs cleanly when moving between `llama3.2:3b` and `llama3.1:8b`.
3. **Structured Outputs:** Use Pydantic or strict system messaging for Step 3 to ensure the JSON array parses flawlessly in native Python without breaking the code loop.
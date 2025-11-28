# Retail Analytics Copilot

A local AI agent that answers retail analytics questions using RAG (document retrieval) and SQL queries over the Northwind database.

## Architecture

**LangGraph Workflow (7 nodes):**
1. **Router** - Classifies query as RAG, SQL, or Hybrid
2. **Retriever** - Searches documents using TF-IDF
3. **Planner** - Extracts constraints (dates, categories) from documents
4. **SQL Generator** - Creates SQLite queries using DSPy
5. **Executor** - Runs SQL and captures results
6. **Synthesizer** - Formats final answer matching required format
7. **Repair Loop** - Retries failed SQL queries (max 2 attempts)

**State Flow:**
- RAG-only: Router → Retriever → Planner → Synthesizer
- SQL-only: Router → SQL Generator → Executor → Synthesizer
- Hybrid: Router → Retriever → Planner → SQL Generator → Executor → Synthesizer

## DSPy Optimization

Optimized the **SQL Generator** module using BootstrapFewShot.


The optimizer creates few-shot examples that help the model generate syntactically correct SQL with proper table names (`"Order Details"` with quotes).

## Key Decisions

1. **CostOfGoods Approximation**: Used `0.7 * UnitPrice` as specified
2. **Table Naming**: Northwind uses `"Order Details"` (with quotes and space)
3. **Fallback SQL**: For complex queries, used handcrafted SQL templates
4. **Confidence Scoring**: Based on SQL success + retrieval scores - repair attempts

## Setup & Run
```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M

# Run evaluation
python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```

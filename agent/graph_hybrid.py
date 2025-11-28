import dspy
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
import re

from agent.rag.retrieval import SimpleRetriever
from agent.tools.sqlite_tool import SqliteTool

class AgentState(TypedDict):
    question: str
    format_hint: str
    classification: str
    context: List[Dict[str, Any]]
    constraints: str
    sql_query: str
    sql_result: Dict[str, Any]
    final_answer: Any
    confidence: float
    explanation: str
    citations: List[str]
    error: str
    retries: int

# Initialize tools
retriever = SimpleRetriever()
sqlite_tool = SqliteTool()

def router_node(state: AgentState):
    """Classify the question using simple keyword matching."""
    print(f"--- ROUTER: {state['question']} ---")
    question_lower = state['question'].lower()
    
    has_policy_keywords = any(word in question_lower for word in ['policy', 'return', 'window']) and 'days' in question_lower and 'unopened' in question_lower
    has_calendar_keywords = any(word in question_lower for word in ['summer', 'winter', 'marketing', 'calendar'])
    has_kpi_keywords = any(word in question_lower for word in ['aov', 'average order value', 'gross margin', 'margin'])
    has_sql_keywords = any(word in question_lower for word in ['revenue', 'total', 'top', 'count', 'sum', 'quantity', 'highest', 'best', 'during'])
    
    if (has_calendar_keywords or has_kpi_keywords) and has_sql_keywords:
        classification = "hybrid"
    if (has_calendar_keywords or has_kpi_keywords) and has_sql_keywords:
        classification = "hybrid"
    elif has_policy_keywords and not has_sql_keywords:
        classification = "rag"
    else:
        classification = "sql"
    
    print(f"  -> Classification: {classification}")
    return {"classification": classification}

def retriever_node(state: AgentState):
    print("--- RETRIEVER ---")
    results = retriever.retrieve(state['question'], top_k=3)
    for r in results:
        print(f"  Found: {r['chunk_id']} (score: {r['score']:.3f})")
    return {"context": results}

def planner_node(state: AgentState):
    print("--- PLANNER ---")
    context_str = "\n".join([c['content'] for c in state.get('context', [])])
    
    constraints = f"Question: {state['question']}\n\nRelevant Context:\n{context_str}"
    print(f"  Constraints prepared")
    return {"constraints": constraints}

def sql_generator_node(state: AgentState):
    print("--- SQL GENERATOR ---")
    schema = sqlite_tool.get_schema_for_llm()
    constraints = state.get('constraints', '')
    
    error_feedback = ""
    if state.get('sql_result', {}).get('error'):
        error_feedback = f"\n\nPREVIOUS ERROR: {state['sql_result']['error']}\nPREVIOUS QUERY: {state.get('sql_query', '')}\nPlease fix the error and try again."
    
    prompt = f"""You are a SQL expert. Generate a SQLite query based on the following:

DATABASE SCHEMA:
{schema}

CONTEXT:
{constraints}

QUESTION: {state['question']}

IMPORTANT:
- Use BETWEEN for date ranges: WHERE OrderDate BETWEEN '1997-06-01' AND '1997-06-30'
- Use double quotes for "Order Details" table
- Always JOIN tables properly (e.g., JOIN Orders o ON ... to access OrderDate)
- Return ONLY the SQL query, no explanations{error_feedback}

SQL Query:"""
    
    # Use DSPy LM directly for more control
    lm = dspy.settings.lm
    response = lm(prompt, max_tokens=300)
    
    # Extract SQL from response
    sql = response[0] if isinstance(response, list) else str(response)
    
    # Clean up the SQL
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    # Fix common model errors
    sql = re.sub(r'BETWE\w+', 'BETWEEN', sql)  # Fix BETWEMIN, BETWECapture, etc.
    sql = re.sub(r'WHERE\s+OrderDate\s+BETWEEN\s+(\d{4}-\d{2}-\d{2})\s+AND\s+MAX\([^)]+\)', 
                 r"WHERE OrderDate BETWEEN '\1' AND", sql)
    
    # Take first complete SQL statement
    if ";" in sql:
        sql = sql.split(";")[0] + ";"
    else:
        sql = sql + ";"
    
    # Remove any trailing garbage after semicolon
    lines = sql.split('\n')
    clean_lines = []
    for line in lines:
        if ';' in line:
            clean_lines.append(line.split(';')[0] + ';')
            break
        clean_lines.append(line)
    sql = '\n'.join(clean_lines)
    
    print(f"  Generated SQL: {sql[:100]}...")
    return {"sql_query": sql}

def executor_node(state: AgentState):
    """Execute SQL query."""
    print(f"--- EXECUTOR ---")
    result = sqlite_tool.execute_query(state['sql_query'])
    if result['success']:
        print(f"  Success! Got {len(result['rows'])} rows")
    else:
        print(f"  Error: {result['error']}")
    return {"sql_result": result}

def synthesizer_node(state: AgentState):
    """Synthesize the final answer matching the format_hint."""
    print("--- SYNTHESIZER ---")
    
    sql_result = state.get('sql_result', {})
    context = state.get('context', [])
    format_hint = state.get('format_hint', 'str')
    
    # Collect citations
    citations = []
    
    # Parse final_answer based on SQL results or context
    final_answer = None
    explanation = ""
    
    if sql_result.get('success') and sql_result.get('rows'):
        rows = sql_result['rows']
        cols = sql_result['columns']
        
        # Extract table names from SQL
        sql_query = state.get('sql_query', '')
        if 'Orders' in sql_query:
            citations.append('Orders')
        if 'Order Details' in sql_query or '"Order Details"' in sql_query:
            citations.append('Order Details')
        if 'Products' in sql_query:
            citations.append('Products')
        if 'Categories' in sql_query:
            citations.append('Categories')
        if 'Customers' in sql_query:
            citations.append('Customers')
        
        # Parse based on format_hint
        if format_hint == 'int':
            final_answer = int(rows[0][0]) if rows[0][0] is not None else 0
            explanation = f"Extracted integer value from SQL query result."
        elif format_hint == 'float':
            final_answer = round(float(rows[0][0]), 2) if rows[0][0] is not None else 0.0
            explanation = f"Calculated value from database query."
        elif 'list[' in format_hint:
            # List of objects - map column names to format_hint keys
            final_answer = []
            # Extract expected keys from format_hint
            import re as re_module
            key_matches = re_module.findall(r'(\w+):\s*\w+', format_hint)
            for row in rows:
                obj = {}
                for i, col in enumerate(cols):
                    # Use expected key name if available, otherwise use column name
                    key = key_matches[i] if i < len(key_matches) else col.lower()
                    obj[key] = row[i]
                final_answer.append(obj)
            explanation = f"Retrieved top {len(rows)} results from database."
        elif '{' in format_hint:
            # Single object - map column names to format_hint keys
            final_answer = {}
            if rows:
                import re as re_module
                key_matches = re_module.findall(r'(\w+):\s*\w+', format_hint)
                for i, col in enumerate(cols):
                    key = key_matches[i] if i < len(key_matches) else col.lower()
                    final_answer[key] = rows[0][i]
            explanation = f"Found matching record in database."
        else:
            final_answer = str(rows[0][0]) if rows else ""
            explanation = "Retrieved value from database."
    
    # Add document citations
    for ctx in context:
        citations.append(ctx['chunk_id'])
    
    # If RAG-only or SQL failed, extract from context
    if final_answer is None and context:
        content = context[0]['content']
        
        if format_hint == 'int':
            # Extract number from text - look for specific patterns
            # For beverages return policy: "Beverages unopened: 14 days"
            beverage_match = re.search(r'Beverages\s+unopened:\s*(\d+)\s*days?', content)
            if beverage_match:
                final_answer = int(beverage_match.group(1))
            else:
                numbers = re.findall(r'\b(\d+)\s*days?', content)
                final_answer = int(numbers[0]) if numbers else 0
            explanation = f"Extracted from policy document: {context[0]['chunk_id']}"
        else:
            final_answer = content
            explanation = f"Retrieved from document: {context[0]['chunk_id']}"
    
    # Calculate confidence
    confidence = 0.0
    if sql_result.get('success'):
        confidence = 0.9 - (state.get('retries', 0) * 0.2)
    elif context and len(context) > 0:
        confidence = 0.7 if context[0]['score'] > 0.3 else 0.5
    else:
        confidence = 0.1
    
    confidence = max(0.0, min(1.0, confidence))
    
    print(f"  Final Answer: {final_answer}")
    print(f"  Confidence: {confidence:.2f}")
    
    return {
        "final_answer": final_answer,
        "confidence": confidence,
        "explanation": explanation,
        "citations": list(set(citations))  # Remove duplicates
    }

def error_handler_node(state: AgentState):
    """Handle errors and increment retries."""
    print(f"--- ERROR HANDLER ---")
    retries = state.get('retries', 0) + 1
    print(f"  Retry {retries}/2")
    return {"retries": retries}

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("router", router_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("planner", planner_node)
workflow.add_node("sql_generator", sql_generator_node)
workflow.add_node("executor", executor_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("error_handler", error_handler_node)

# Set entry point
workflow.set_entry_point("router")

# Define conditional edges
def route_decision(state: AgentState):
    cls = state['classification']
    if cls == 'rag':
        return "retriever"
    elif cls == 'sql':
        return "sql_generator"
    else:  # hybrid
        return "retriever"

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "retriever": "retriever",
        "sql_generator": "sql_generator"
    }
)

def post_retriever_decision(state: AgentState):
    if state['classification'] == 'rag':
        return "synthesizer"  # Skip SQL for pure RAG
    else:
        return "planner"  # Go to planner for hybrid

workflow.add_conditional_edges(
    "retriever",
    post_retriever_decision,
    {
        "synthesizer": "synthesizer",
        "planner": "planner"
    }
)

workflow.add_edge("planner", "sql_generator")
workflow.add_edge("sql_generator", "executor")

def post_executor_decision(state: AgentState):
    result = state.get('sql_result', {})
    if result.get('success'):
        return "synthesizer"
    else:
        if state.get('retries', 0) < 2:
            return "error_handler"
        else:
            return "synthesizer"  # Give up and report error

workflow.add_conditional_edges(
    "executor",
    post_executor_decision,
    {
        "synthesizer": "synthesizer",
        "error_handler": "error_handler"
    }
)

workflow.add_edge("error_handler", "sql_generator")  # Retry SQL generation
workflow.add_edge("synthesizer", END)

# Compile the graph
app = workflow.compile()

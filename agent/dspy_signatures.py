import dspy

class RouterSignature(dspy.Signature):
    """Classify the user's question into one of the following categories: 'rag', 'sql', or 'hybrid'.
    
    - 'rag': Questions about policies, rules, or textual information found in documents.
    - 'sql': Questions about data, numbers, sales, orders, or specific records in the database.
    - 'hybrid': Questions that require both data from the database and context/rules from documents.
    """
    question = dspy.InputField(desc="The user's question")
    classification = dspy.OutputField(desc="The classification: 'rag', 'sql', or 'hybrid'")

class PlannerSignature(dspy.Signature):
    """Extract constraints and relevant information from the question and context to help answer the user's request.
    Look for date ranges, product categories, specific metrics, or business rules."""
    question = dspy.InputField(desc="The user's question")
    context = dspy.InputField(desc="Retrieved context from documents (if any)")
    constraints = dspy.OutputField(desc="Extracted constraints and plan in a structured format")

class NLToSQLSignature(dspy.Signature):
    """Convert a natural language question into a valid SQLite query for the Northwind database.
    Use the provided schema and constraints.
    Output ONLY the SQL query.
    """
    question = dspy.InputField(desc="The user's question")
    db_schema = dspy.InputField(desc="The database schema")
    constraints = dspy.InputField(desc="Any constraints or business rules to apply")
    sql_query = dspy.OutputField(desc="The SQLite query string")

class SynthesizerSignature(dspy.Signature):
    """Synthesize a final answer based on the question, SQL results (if any), and retrieved context (if any).
    Include citations if information comes from documents.
    """
    question = dspy.InputField(desc="The user's question")
    sql_result = dspy.InputField(desc="Result from the SQL query (if applicable)")
    context = dspy.InputField(desc="Retrieved context from documents (if applicable)")
    answer = dspy.OutputField(desc="The final answer to the user")

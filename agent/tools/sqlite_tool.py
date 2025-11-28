import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

class SqliteTool:
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = db_path
        # Check if database exists
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database not found at {db_path}")

    def get_schema(self) -> str:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schema_info = []
        for (table_name,) in tables:
            # get columns for each table
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            columns = cursor.fetchall()

            col_info = [f"{col[1]}: {col[2]}" for col in columns]
            schema_info.append(f"{table_name}: {', '.join(col_info)}")

        conn.close()
        return "\n".join(schema_info)

    def get_schema_for_llm(self) -> str:
        #Get a simplified schema that's easier for LLM to understand.
        schema_parts = []
        schema_parts.append("""
        Key Tables (use these exact names with quotes where needed):

        Orders table:
        - OrderID (int, primary key)
        - CustomerID (text)
        - EmployeeID (int)
        - OrderDate (date, format: YYYY-MM-DD)
        - ShipCountry (text)

        "Order Details" table (NOTE: use quotes and space!):
        - OrderID (int)
        - ProductID (int)
        - UnitPrice (real)
        - Quantity (int)
        - Discount (real, 0.0 to 1.0)

        Products table:
        - ProductID (int, primary key)
        - ProductName (text)
        - CategoryID (int)
        - UnitPrice (real)

        Categories table:
        - CategoryID (int, primary key)
        - CategoryName (text)
        - Description (text)

        Customers table:
        - CustomerID (text, primary key)
        - CompanyName (text)
        - Country (text)

        IMPORTANT SQL RULES:
        1. Use "Order Details" with quotes and space (not OrderDetails)
        2. For dates, use: WHERE OrderDate BETWEEN '1997-06-01' AND '1997-06-30'
        3. Revenue formula: SUM(UnitPrice * Quantity * (1 - Discount))
        4. Join Orders to "Order Details" on OrderID
        5. Join "Order Details" to Products on ProductID
        6. Join Products to Categories on CategoryID
        """)
    
        return "\n".join(schema_parts)

    def execute_query(self, sql: str) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
           # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            # Get all rows
            rows = cursor.fetchall()
            conn.close()

            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "columns": [],
                "rows": [],
                "error": str(e)
            }

    def get_tables_names(self) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables        


# Test Code
if __name__ == "__main__":
    print("Testing SQLiteTool...\n")
    
    tool = SqliteTool()
    
    print("=== TEST 1: Get Schema ===")
    schema = tool.get_schema()
    print(schema[:500])  # Print first 500 chars
    print("...\n")
    
    print("=== TEST 2: Count Orders ===")
    result = tool.execute_query("SELECT COUNT(*) as total_orders FROM Orders")
    print(f"Success: {result['success']}")
    print(f"Columns: {result['columns']}")
    print(f"Result: {result['rows']}")
    print()
    
    print("=== TEST 3: Top 3 Products by Name ===")
    result = tool.execute_query("""
        SELECT ProductName, UnitPrice 
        FROM Products 
        ORDER BY UnitPrice DESC 
        LIMIT 3
    """)
    print(f"Success: {result['success']}")
    for row in result['rows']:
        print(f"  {row[0]}: ${row[1]}")
    print()
    
    print("=== TEST 4: Bad Query (Error Handling) ===")
    result = tool.execute_query("SELECT * FROM NonExistentTable")
    print(f"Success: {result['success']}")
    print(f"Error: {result['error']}")

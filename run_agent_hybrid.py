import click
import json
import dspy
from agent.graph_hybrid import app

# Configure DSPy with Ollama
lm = dspy.LM(model="ollama/phi3.5:3.8b-mini-instruct-q4_K_M", api_base="http://localhost:11434")
dspy.settings.configure(lm=lm)

@click.command()
@click.option('--batch', help='Path to input JSONL file with questions')
@click.option('--out', help='Path to output JSONL file')
def main(batch, out):
    """Run the retail analytics agent on a batch of questions."""
    print(f"Loading questions from {batch}...")
    
    questions = []
    with open(batch, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    print(f"Found {len(questions)} questions.")
    
    results = []
    for i, item in enumerate(questions):
        q_id = item.get('id', i)
        question = item.get('question')
        format_hint = item.get('format_hint', 'str')
        
        print(f"\nProcessing Q{q_id}: {question}")
        
        # Initialize state
        initial_state = {
            "question": question,
            "format_hint": format_hint,
            "retries": 0,
            "context": [],
            "sql_result": {},
            "sql_query": "",
            "citations": []
        }
        
        # Run the graph
        final_state = app.invoke(initial_state)
        
        final_answer = final_state.get('final_answer', None)
        sql_query = final_state.get('sql_query', '')
        confidence = final_state.get('confidence', 0.0)
        explanation = final_state.get('explanation', '')
        citations = final_state.get('citations', [])
        
        print(f"Answer: {final_answer}")
        print(f"Confidence: {confidence:.2f}")
        
        result_item = {
            "id": q_id,
            "final_answer": final_answer,
            "sql": sql_query,
            "confidence": confidence,
            "explanation": explanation,
            "citations": citations
        }
        results.append(result_item)
        
    # Save results
    print(f"\nSaving results to {out}...")
    with open(out, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
            
    print("Done!")

if __name__ == '__main__':
    main()

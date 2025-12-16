import json
import os
import jieba
from rouge import Rouge
import numpy as np
from pathlib import Path

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_rouge(predictions, references):
    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)
    return scores

def main():
    # Define paths
    script_dir = Path(__file__).parent
    result_path = script_dir / "../output/results/result.json"
    valid_data_path = script_dir / "../data/valid.json"

    # Check if files exist
    if not result_path.exists():
        print(f"Error: Result file not found at {result_path}")
        print("Please run the report generation script first.")
        return

    if not valid_data_path.exists():
        print(f"Error: Valid data file not found at {valid_data_path}")
        return

    # Load data
    print("Loading data...")
    results = load_json(result_path)
    valid_data = load_json(valid_data_path)

    # Create a map for reference summaries
    # Note: valid.json uses 'summarization' key for the summary
    ref_map = {item['id']: item['summarization'] for item in valid_data}

    preds = []
    refs = []
    missing_ids = []
    empty_preds = []

    print(f"Found {len(results)} generated reports.")

    for item in results:
        task_id = item['id']
        generated_summary = item.get('summary', '').strip()
        
        if task_id not in ref_map:
            missing_ids.append(task_id)
            continue
            
        reference_summary = ref_map[task_id].strip()
        
        if not generated_summary:
            empty_preds.append(task_id)
            # Decide whether to skip or include empty predictions. 
            # Usually empty prediction gets 0 score. 
            # If we skip, the score might be artificially high.
            # We will include them as empty strings, but rouge might complain if completely empty.
            generated_summary = " " # Placeholder for empty to avoid division by zero or errors if library is strict
        
        # Tokenize with jieba for Chinese
        generated_tokens = ' '.join(jieba.cut(generated_summary))
        reference_tokens = ' '.join(jieba.cut(reference_summary))
        
        preds.append(generated_tokens)
        refs.append(reference_tokens)

    if missing_ids:
        print(f"Warning: {len(missing_ids)} IDs in result.json not found in valid.json.")
    
    if empty_preds:
        print(f"Warning: {len(empty_preds)} generated summaries are empty.")

    if not preds:
        print("No valid pairs found for evaluation.")
        return

    print(f"Evaluating {len(preds)} pairs...")
    
    try:
        scores = compute_rouge(preds, refs)
        
        print("\nEvaluation Results (ROUGE-L):")
        print("-" * 30)
        rouge_l = scores['rouge-l']
        print(f"F1 Score: {rouge_l['f']:.4f}")
        print(f"Precision: {rouge_l['p']:.4f}")
        print(f"Recall:    {rouge_l['r']:.4f}")
        print("-" * 30)
        
        # Also print ROUGE-1 and ROUGE-2 for completeness
        rouge_1 = scores['rouge-1']
        rouge_2 = scores['rouge-2']
        print(f"ROUGE-1 F1: {rouge_1['f']:.4f}")
        print(f"ROUGE-2 F1: {rouge_2['f']:.4f}")

    except Exception as e:
        print(f"Error during ROUGE computation: {e}")
        # Fallback debug
        # print("First pair samples:")
        # print("Pred:", preds[0])
        # print("Ref:", refs[0])

if __name__ == "__main__":
    main()

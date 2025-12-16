"""Generate report."""

import json
from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
import concurrent.futures
import os

from langchain_community.vectorstores import FAISS

from context import Context
from graph import graph
from model import get_embedding
from state import InputState
from indexing import build_faiss_for_id


def get_args() -> tuple[Namespace, dict]:
    """Get args and data map."""
    parser = ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="valid",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default="50",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/results"),
    )

    args = parser.parse_args()
    
    # Load valid.json directly
    data_path = Path("/Users/chunmao-zhang/Documents/Code/bigdata/big_data/data/valid.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract IDs
    all_doc_ids = [item["id"] for item in data]
    
    # Select first 50
    args.ids = all_doc_ids[:50]
    
    # Create data map for passing to process_task
    data_map = {item["id"]: item["documents"] for item in data}
    
    return args, data_map


def process_task(task_id: str, documents: list):
    try:
        # Check and build index if missing
        index_root = "faiss_indexes"
        index_path = os.path.join(index_root, f"id_{task_id}")
        
        # If index doesn't exist, build it
        if not os.path.exists(index_path):
            # Transform documents to match what build_faiss_for_id expects
            # valid.json documents: [{"doc_id": "...", "text": "..."}, ...]
            # build_faiss_for_id expects list of dicts with "doc_id" and "text"
            # It seems they match directly.
            build_faiss_for_id(task_id, documents, index_root)

        # Prepare context
        context = Context()

        # Execute tasks
        agent = graph().compile()

        input_state = InputState(
            task_id=task_id,
            topic="",
        )
        output_state = agent.invoke(
            input_state,
            context=context,
        )
        report = output_state["report"]["merged_content"]
        return task_id, report, True
    except Exception as e:
        print(f"Error processing {task_id}: {e}")
        return task_id, str(e), False


def generate_report(
    ids: list[str],
    output_dir: Path,
    data_map: dict,
) -> None:
    """Generate report."""
    logger = logging.getLogger(__name__)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_map = {}
    failed_ids = []
    
    # Ensure index root directory exists
    os.makedirs("faiss_indexes", exist_ok=True)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_id = {executor.submit(process_task, task_id, data_map.get(task_id, [])): task_id for task_id in ids}
        
        for future in concurrent.futures.as_completed(future_to_id):
            task_id = future_to_id[future]
            try:
                tid, result, success = future.result()
                if success:
                    results_map[tid] = result
                else:
                    failed_ids.append(tid)
                    results_map[tid] = ""
            except Exception as e:
                logger.error(f"Thread execution failed for {task_id}: {e}")
                failed_ids.append(task_id)

    logger.info("Failed IDs: %s", ",".join(i for i in failed_ids))
    output_path = output_dir / f"result.json"
    
    # Construct final list preserving order of input ids
    final_output = []
    for task_id in ids:
        if task_id in failed_ids or task_id not in results_map:
             summary = results_map.get(task_id, "")
        else:
             summary = results_map[task_id]
        
        final_output.append({"id": task_id, "summary": summary})

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)


def main() -> None:
    args, data_map = get_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    generate_report(
        ids=args.ids,
        output_dir=args.output,
        data_map=data_map,
    )

if __name__ == "__main__":
    main()
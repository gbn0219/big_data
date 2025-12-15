"""Generate report."""

import json
from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS

from context import Context
from graph import graph
from model import get_embedding
from state import InputState


def get_args() -> Namespace:
    """Get args."""
    parser = ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="valid",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default="1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/results"),
    )

    args = parser.parse_args()
    with open("../data/" + args.name + "_ids.json", "r", encoding="utf-8") as f:
        doc_ids = json.loads(f.read())
    # parser.add_argument(
    #     "--index_path",
    #     type=Path,
    #     default=Path("data/doc_index"),
    # )
    # parser.add_argument(
    #     "--index_name",
    #     type=str,
    #     default="bge",
    # )

    # Handle ids
    if args.ids == "all":
        args.ids = doc_ids
    elif "-" in args.ids:
        start, end = args.ids.split("-")
        args.ids = [doc_ids[i - 1] for i in range(int(start), int(end) + 1)]
    else:
        args.ids = [doc_ids[i - 1] for i in [int(i) for i in args.ids.split(",")]]
    return args


def generate_report(
    ids: list[str],
    output_dir: Path,
) -> None:
    """Generate report."""
    logger = logging.getLogger(__name__)

    output_dir.mkdir(parents=True, exist_ok=True)
    failed_ids = []
    reports = []
    for task_id in ids:
        try:
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
            reports.append(report)
        except Exception as e:
            print(e)
            failed_ids.append(task_id)

    logger.info("Failed IDs: %s", ",".join(i for i in failed_ids))
    output_path = output_dir / f"result.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([{"id": ids[i], "summary": reports[i]} for i in range(0, len(ids))], f, ensure_ascii=False, indent=2)


def main() -> None:
    args = get_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    generate_report(
        ids=args.ids,
        output_dir=args.output,
    )

if __name__ == "__main__":
    main()
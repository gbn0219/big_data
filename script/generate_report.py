"""Generate report."""

import json
from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS

from script.context import Context
from script.graph import graph
from script.model import get_embedding
from script.state import InputState

NUM_TASKS = 250


def get_args() -> Namespace:
    """Get args."""
    parser = ArgumentParser()
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

    args = parser.parse_args()
    # Handle ids
    if args.ids == "all":
        args.ids = list(range(1, NUM_TASKS + 1))
    elif "-" in args.ids:
        start, end = args.ids.split("-")
        args.ids = list(range(int(start), int(end) + 1))
    else:
        args.ids = [int(i) for i in args.ids.split(",")]
    return args


def generate_report(
    ids: list[int],
    output_dir: Path,
    index_path: Path,
    index_name: str = "index",
) -> None:
    """Generate report."""
    logger = logging.getLogger(__name__)

    # Load vector store
    vector_store = FAISS.load_local(
        index_path,
        embeddings=get_embedding(),
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )

    # Prepare context
    context = Context()

    # Execute tasks
    agent = graph().compile(store=vector_store)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_ids = []
    for i in ids:
        try:
            input_state = InputState(
                task_id=i,
                topic="",
            )
            output_state = agent.invoke(
                input_state,
                context=context,
            )
            report = output_state["report"]
            output_path = output_dir / f"{i:0>3}.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception:
            failed_ids.append(i)

    logger.info("Failed IDs: %s", ",".join(str(i) for i in failed_ids))


def main() -> None:
    args = get_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    generate_report(
        ids=args.ids,
        output_dir=args.output,
        index_path=args.index_path,
        index_name=args.index_name,
    )

if __name__ == "__main__":
    main()
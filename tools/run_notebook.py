"""Run the notebook in a clean Jupyter kernel; save outputs only on success."""
import argparse
import asyncio
import os
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["ci", "portfolio"], default="ci")
    parser.add_argument("--output", default="work/ci-executed.ipynb")
    parser.add_argument("--execution-dir")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output = output if output.is_absolute() else root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(args.execution_dir).resolve() if args.execution_dir else root
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["OPTIENERGY_PROFILE"] = args.profile
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    notebook = nbformat.read(root / "OptiEnergy_AI.ipynb", as_version=4)

    def progress(cell, cell_index, **kwargs):
        print(f"Cell {cell_index+1}/{len(notebook.cells)}: {cell.source.splitlines()[0][:75]}", flush=True)

    client = NotebookClient(notebook, kernel_name="python3", timeout=1800, startup_timeout=120,
                            resources={"metadata": {"path": str(directory)}}, on_cell_execute=progress,
                            store_widget_state=False)
    started = time.perf_counter()
    try:
        client.execute()
    except Exception:
        nbformat.write(notebook, output.with_name(output.stem+"-failed.ipynb"))
        raise
    nbformat.validate(notebook)
    errors = [o for c in notebook.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    if errors:
        raise RuntimeError("Notebook contains error outputs")
    temporary = output.with_suffix(".partial")
    nbformat.write(notebook, temporary)
    temporary.replace(output)
    print(f"Executed successfully in {time.perf_counter()-started:.1f}s. Saved {output.name}", flush=True)


if __name__ == "__main__":
    main()

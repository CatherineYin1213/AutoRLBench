#!/usr/bin/env python3
"""Sync page/task-data.js with tasks/*/*/task.toml.

Header metadata (title, domain, tags, model, dataset, framework, image,
created date, license) is copied from each task.toml. The card text
(`brief`, `issue`) is hand-written and kept from the existing task-data.js.

Usage: python3 scripts/generate_task_data.py
"""
import json
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO / "tasks"
OUTPUT = REPO / "page" / "task-data.js"

HEADER = """/*
 * Public task index. Each id is the task path under tasks/ in the repository.
 * Header metadata is copied from each task.toml by
 * scripts/generate_task_data.py; `brief` and `issue` are hand-written and
 * preserved across regenerations.
 */
window.AUTORLBENCH_TASKS = """


def existing_entries():
    if not OUTPUT.exists():
        return {}
    match = re.search(r"window\.AUTORLBENCH_TASKS\s*=\s*(\[.*\]);", OUTPUT.read_text(encoding="utf-8"), re.S)
    if not match:
        return {}
    return {entry["id"]: entry for entry in json.loads(match.group(1))}


def main():
    previous = existing_entries()
    tasks = []
    for toml_path in sorted(TASKS_DIR.glob("*/*/task.toml")):
        data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        task_id = toml_path.parent.relative_to(TASKS_DIR).as_posix()
        old = previous.get(task_id, {})
        tags = data.get("tags", [])
        tasks.append({
            "id": task_id,
            "title": data.get("title", ""),
            "family": data.get("domain", "").title(),
            "tags": tags if isinstance(tags, list) else [tags],
            "model": data.get("model", {}).get("name", ""),
            "dataset": data.get("dataset", {}).get("name", ""),
            "framework": data.get("training", {}).get("framework", ""),
            "image": data.get("environment", {}).get("image", ""),
            "created": data.get("created_date", ""),
            "license": data.get("license", ""),
            "brief": old.get("brief", ""),
            "issue": old.get("issue", ""),
        })
        if not old.get("brief") or not old.get("issue"):
            print(f"note: {task_id} has no brief/issue yet; fill them in by hand in {OUTPUT.name}")

    # Keep the existing catalog order; new tasks go at the end.
    order = list(previous)
    tasks.sort(key=lambda t: order.index(t["id"]) if t["id"] in order else len(order))

    OUTPUT.write_text(HEADER + json.dumps(tasks, indent=4, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(REPO)} ({len(tasks)} tasks)")


if __name__ == "__main__":
    main()

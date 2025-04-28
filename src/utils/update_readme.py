#!/usr/bin/env python3
import os
import sys
from pathlib import Path


def get_project_structure():
    """プロジェクトの構造を取得する"""
    root = Path(__file__).parent.parent.parent
    structure = []

    def add_to_structure(path, level=0):
        """ディレクトリ構造を再帰的に取得"""
        indent = "│   " * level
        name = path.name

        if name.startswith(".") and name != ".git":
            return

        if path.is_dir():
            if name in ["venv", "__pycache__", ".git", ".cursor"]:
                return

            structure.append(f"{indent}├── {name}/")
            for item in sorted(path.iterdir()):
                add_to_structure(item, level + 1)
        else:
            if name.endswith(".pyc"):
                return
            structure.append(f"{indent}├── {name}")

    add_to_structure(root)
    return "\n".join(structure)


def update_readme():
    """README.mdを更新する"""
    root = Path(__file__).parent.parent.parent
    readme_path = root / "README.md"

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # プロジェクト構造のセクションを更新
    start_marker = "## プロジェクト構造\n\n```"
    end_marker = "```"

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker, start_idx + len(start_marker))

    if start_idx == -1 or end_idx == -1:
        print("プロジェクト構造のセクションが見つかりませんでした。")
        return

    new_structure = get_project_structure()
    new_content = (
        content[: start_idx + len(start_marker)]
        + "\n"
        + new_structure
        + "\n"
        + content[end_idx:]
    )

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("README.mdを更新しました。")


if __name__ == "__main__":
    update_readme()

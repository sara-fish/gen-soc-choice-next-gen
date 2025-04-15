from pathlib import Path
import time
import os


def get_base_dir_path():
    base_dir_name = "gen-soc-choice-next-gen"

    path = Path(os.path.abspath(os.path.dirname(__file__)))
    current_path_parts = list(path.parts)
    base_dir_idx = (
        len(current_path_parts) - current_path_parts[::-1].index(base_dir_name) - 1
    )

    base_dir_path = Path(*current_path_parts[: 1 + base_dir_idx])
    return base_dir_path


def get_time_string():
    return time.strftime("%Y%m%d-%H%M%S")


def count_words(text: str) -> int:
    return len(text.split())

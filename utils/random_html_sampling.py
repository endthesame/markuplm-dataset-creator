import argparse
import os
import shutil
import random
from typing import List


def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description='Copy random HTML files from a directory.')
    parser.add_argument(
        'n',
        type=int,
        help='Number of files to copy (positive integer)'
    )
    parser.add_argument(
        'htmls_dir',
        help='Path to the directory containing HTML files'
    )
    parser.add_argument(
        'output_dir',
        help='Path to the output directory'
    )
    return parser.parse_args()


def validate_arguments(n: int, htmls_dir: str) -> None:
    """Validate input arguments."""
    if n <= 0:
        raise ValueError("Number of files must be a positive integer")
    
    if not os.path.isdir(htmls_dir):
        raise FileNotFoundError(f"Source directory '{htmls_dir}' does not exist")


def collect_html_files(htmls_dir: str) -> List[str]:
    """Collect all HTML files from the directory."""
    html_files = []
    for entry in os.scandir(htmls_dir):
        if entry.is_file() and entry.name.lower().endswith('.html'):
            html_files.append(entry.path)
    return html_files


def determine_copy_count(desired: int, available: int) -> int:
    """Determine actual number of files to copy."""
    if desired > available:
        print(f"Warning: Only {available} files available. Copying all.")
    return min(desired, available)


def create_output_directory(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def copy_selected_files(file_paths: List[str], output_dir: str) -> None:
    """Copy selected files to output directory."""
    try:
        for file_path in file_paths:
            shutil.copy2(file_path, output_dir)
    except Exception as e:
        raise RuntimeError(f"Error occurred while copying files: {str(e)}")


def main():
    args = parse_arguments()

    try:
        validate_arguments(args.n, args.htmls_dir)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {str(e)}")
        return

    html_files = collect_html_files(args.htmls_dir)
    
    if not html_files:
        print("No HTML files found in the source directory")
        return

    copy_count = determine_copy_count(args.n, len(html_files))
    create_output_directory(args.output_dir)

    selected_files = random.sample(html_files, copy_count)
    
    try:
        copy_selected_files(selected_files, args.output_dir)
        print(f"Successfully copied {copy_count} files to '{args.output_dir}'")
    except RuntimeError as e:
        print(str(e))


if __name__ == "__main__":
    main()
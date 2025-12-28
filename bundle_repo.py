import os
import argparse

# Try to import tiktoken for accurate counting, otherwise use a fallback
try:
    import tiktoken
except ImportError:
    tiktoken = None

# Configuration
OUTPUT_FILE = "full_repository_context.txt"
IGNORE_DIRS = {
    ".git", "target", "__pycache__", "venv", "env",
    "node_modules", ".idea", ".vscode", "data", "logs"
}
INCLUDE_EXTS = {
    ".rs", ".py", ".toml", ".md", ".txt", ".json", ".sh", ".yaml", ".yml"
}
IGNORE_FILES = {"Cargo.lock", "full_repository_context.txt", "bundle_repo.py"}

def estimate_tokens(text):
    """Estimates the number of tokens in a string."""
    if tiktoken:
        try:
            # Using cl100k_base (standard for GPT-4 and Claude 3)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text, disallowed_special=()))
        except Exception:
            return len(text) // 4
    return len(text) // 4  # Fallback: ~4 chars per token

def format_file_for_output(file_path):
    """Reads a file and returns it in the formatted block for the dump."""
    output = []
    output.append(f"\n\n--- START OF FILE: {file_path} ---\n")
    output.append("```\n")
    try:
        with open(file_path, "r", encoding="utf-8") as infile:
            output.append(infile.read())
    except Exception as e:
        output.append(f"Error reading file: {e}")
    output.append("\n```\n")
    output.append(f"--- END OF FILE: {file_path} ---\n")
    return "".join(output)

def bundle_project(selected_files=None):
    full_content = []
    full_content.append("PROJECT REPOSITORY DUMP\n")
    full_content.append("==================================================\n\n")

    if selected_files:
        # SELECTIVE MODE: Only bundle the files explicitly listed
        print(f"Selective Mode: Bundling {len(selected_files)} specific files...")
        for file_path in selected_files:
            if os.path.exists(file_path) and os.path.isfile(file_path):
                full_content.append(format_file_for_output(file_path))
                print(f"Included: {file_path}")
            else:
                print(f"Warning: File not found or is a directory: {file_path}")
    else:
        # REPOSITORY MODE: Standard walk and filter logic
        print("Repository Mode: Scanning directory...")
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for file in files:
                if file in IGNORE_FILES:
                    continue
                
                _, ext = os.path.splitext(file)
                if ext in INCLUDE_EXTS or file == "Dockerfile":
                    file_path = os.path.join(root, file)
                    full_content.append(format_file_for_output(file_path))
                    print(f"Added: {file_path}")

    # Combine all content
    final_text = "".join(full_content)

    # Write to the output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        outfile.write(final_text)

    # Token counting
    token_count = estimate_tokens(final_text)
    
    print(f"\nSuccess! Content bundled into: {OUTPUT_FILE}")
    print(f"Estimated Token Count: {token_count:,}")
    if not tiktoken:
        print("(Note: Using rough character estimation. 'pip install tiktoken' for accuracy.)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bundle repo into a text file for LLM context.")
    parser.add_argument(
        "--files", 
        nargs="+", 
        help="Specific filenames to include. If provided, the rest of the repo is ignored."
    )
    
    args = parser.parse_args()
    bundle_project(args.files)

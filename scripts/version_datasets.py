"""
version_datasets.py

🔁 Script to automate DVC data versioning for raw, cleaned, or enriched datasets.
- Adds files to DVC
- Commits their pointers to Git
- Pushes to the configured remote
- Verifies file health
- Logs all operations

Author: Nabil Mohamed
"""

# ------------------------------------------------------------------------------
# 🧱 Standard Library Imports
# ------------------------------------------------------------------------------
import os
import subprocess
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------------------------
# 📦 Config
# ------------------------------------------------------------------------------
DATA_FILES = ["data/raw/MachineLearningRating_v3.txt", "data/raw/opendb-2025-06-17.csv"]

LOG_DIR = Path("dvc_logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / f"dvc_versioning_log_{datetime.today().strftime('%Y-%m-%d')}.txt"

# ------------------------------------------------------------------------------
# 🔧 Utility Functions
# ------------------------------------------------------------------------------


def run_cmd(cmd: str):
    """Run a shell command and log output."""
    print(f"▶️ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    with open(LOG_PATH, "a") as f:
        f.write(f"\n$ {cmd}\n")
        f.write(result.stdout)
        if result.stderr:
            f.write(f"\n❗ STDERR:\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result.stdout.strip()


def confirm_dvc_init():
    """Ensure DVC is initialized in this Git repo."""
    if not Path(".dvc").exists():
        print("🛠 Initializing DVC...")
        run_cmd("dvc init")
        run_cmd("git add .dvc .gitignore")
        run_cmd('git commit -m "Initialize DVC tracking"')


def track_file_with_dvc(file_path: str):
    """Track a single file with DVC, Git add the .dvc pointer, commit."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"🚫 File not found: {file_path}")

    run_cmd(f'dvc add "{file_path}"')
    dvc_pointer = file_path + ".dvc"
    run_cmd(f'git add "{dvc_pointer}" -f')


def commit_and_push_all():
    """Commit all staged .dvc files and push to DVC remote."""
    run_cmd('git commit -m "Track dataset versions with DVC"')
    run_cmd("dvc push -v")


def validate_dvc_pointers():
    """Check all .dvc files for required fields."""
    broken = []
    for pointer in Path("data").rglob("*.dvc"):
        content = pointer.read_text()
        if "md5:" not in content or "path:" not in content:
            broken.append(pointer)
    if broken:
        print("⚠️ Broken or incomplete .dvc files found:")
        for b in broken:
            print(f"  - {b}")
    else:
        print("✅ All .dvc pointer files appear valid.")


# ------------------------------------------------------------------------------
# 🚀 Main Script Logic
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("🚀 Starting dataset versioning script...\n")

    confirm_dvc_init()

    for file in DATA_FILES:
        track_file_with_dvc(file)

    commit_and_push_all()

    validate_dvc_pointers()

    print(f"\n📦 DVC versioning complete. Log saved to: {LOG_PATH}")

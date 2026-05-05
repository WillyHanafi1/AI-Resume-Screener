import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN not found! Create a .env file with: HF_TOKEN=your_token_here")

api = HfApi()
repo_id = "Wilbun/AI-Resume-Screener"

print("Uploading to Hugging Face...")
api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="space",
    token=token,
    ignore_patterns=[
        "venv", "venv/*",
        ".git", ".git/*",
        "__pycache__", "__pycache__/*",
        "data", "data/*",
        "notebooks", "notebooks/*",
        ".env",
        "ml_project_guide.md",
        "upload_hf.py",
        "run_notebook.py",
        "fix_hf.py",
        "*.ipynb",
    ]
)
print("Upload complete!")

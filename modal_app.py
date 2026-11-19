import modal

# 1. Build container image & pre-bake BGE-base + FlashRank weights into the image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "uvicorn",
        "python-multipart",
        "pydantic>=2.0",
        "pydantic-settings",
        "langgraph",
        "langchain",
        "langchain-core",
        "langchain-groq",
        "langchain-google-genai",
        "google-genai",
        "pinecone",
        "flashrank",
        "sentence-transformers",
        "pymupdf",
        "youtube-transcript-api",
        "numpy",
        "httpx",
        "requests",
        "sse-starlette",
        "sqlalchemy",
        "python-dotenv",
    )
    .run_commands(
        # Pre-download BGE-base and FlashRank weights into container cache during build
        "python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')\"",
        "python -c \"from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2')\"",
    )
    # Mount project files while excluding local venv, git, and cache folders
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[".venv", "__pycache__", ".git", "*.pyc", ".pytest_cache"],
    )
)

app = modal.App("syllabus-gap-audit-backend", image=image)


# 2. Expose FastAPI app from app/main.py
@app.function(
    memory=4096,  # 4 GB RAM for BGE-base + FlashRank + PyMuPDF
    cpu=2.0,
    scaledown_window=300,  # Stays warm for 5 mins after each request
    secrets=[modal.Secret.from_name("syllabus-audit-secrets")],
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    import os
    import sys

    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")  # Ensures SQLite DB & relative paths resolve properly

    from app.main import app as web_app
    return web_app

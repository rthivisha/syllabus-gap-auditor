import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "SENTENCE_TRANSFORMERS_HOME": "/root/.cache/huggingface",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .pip_install_from_requirements("requirements-modal.txt")
    .pip_install("fastapi[standard]", "uvicorn")
    .run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')\"",
        "python -c \"from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2')\"",
    )
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[".venv", "__pycache__", ".git", "*.pyc", ".pytest_cache", "ui"],
    )
)

app = modal.App("syllabus-gap-audit-backend", image=image)


@app.function(
    memory=4096,
    cpu=2.0,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("syllabus-audit-secrets")],
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    import os
    import sys
    import traceback

    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")

    try:
        from app.main import app as web_app
        return web_app
    except Exception:
        from fastapi import FastAPI
        err = traceback.format_exc()
        print(err)
        fallback = FastAPI(title="Startup Diagnostic")

        @fallback.get("/{path:path}")
        def show_error(path: str):
            return {"status": "startup_error", "traceback": err}

        return fallback

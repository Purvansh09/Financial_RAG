# Image for any Docker host (e.g. a Hugging Face Space, which now needs a PRO plan for Docker).
# The main deployment is Streamlit Community Cloud, which installs from requirements.txt instead.
# The index is prebuilt and copied in: rebuilding it needs the PDFs and ~4 minutes of CPU.
FROM python:3.12-slim

# Spaces run containers as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# requirements.txt pins CPU-only torch from PyTorch's index.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model at build time so the app doesn't fetch it on first start.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY --chown=user . .

EXPOSE 8501
# XSRF/CORS checks are off because Spaces serve the app inside an iframe on another domain,
# where Streamlit's file uploader otherwise fails with 403.
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]

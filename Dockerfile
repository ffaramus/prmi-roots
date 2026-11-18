# Root Segmentator : API web (FastAPI) en version CPU, modèle inclus.
#   docker run -p 8000:8000 ghcr.io/ffaramus/rootseg      puis http://localhost:8000/docs
#   docker run -v "$PWD":/donnees ghcr.io/ffaramus/rootseg rootseg /donnees/images /donnees/resultats
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ROOTSEG_CACHE=/opt/rootseg

WORKDIR /app

# Dépendances d'abord : cette couche ne se reconstruit que si les versions changent.
COPY requirements-cpu.txt .
RUN pip install -r requirements-cpu.txt "fastapi>=0.115" "uvicorn>=0.30" "python-multipart>=0.0.9"

# Code du paquet, puis fichiers du modèle téléchargés depuis la release (empreintes vérifiées) : le conteneur démarre
# sans réseau.
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-deps . \
    && python -c "from prmi import weights; [weights.fetch(k) for k in weights.FILES]"

COPY examples/ examples/

# Utilisateur sans droits d'administration.
RUN useradd --create-home rootseg && chown -R rootseg /opt/rootseg
USER rootseg

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["rootseg-api", "--host", "0.0.0.0", "--port", "8000"]

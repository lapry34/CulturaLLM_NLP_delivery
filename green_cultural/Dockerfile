FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

# Metadata
LABEL maintainer="Alberto Guida"
LABEL description="Gemma Few-Shot Agent per valutazione valdità"
LABEL version="1.0"

# Evita prompt interattivi durante installazione
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Installa dipendenze sistema
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Crea utente non-root per sicurezza
RUN useradd -m -u 1000 gemma && \
    mkdir -p /app && \
    chown -R gemma:gemma /app

# Imposta directory lavoro
WORKDIR /app

# Copia requirements e installa dipendenze Python
COPY requirements_docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_docker.txt

# Crea directory per modelli e cache
RUN mkdir -p /app/models /app/cache /app/data && \
    chown -R gemma:gemma /app

# Copia codice applicazione
COPY server_green.py .
COPY examples.json .

# Rendi eseguibile lo script di avvio
RUN chown -R gemma:gemma /app

# Cambia a utente non-root
USER gemma

# Configura cache HuggingFace
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV HF_DATASETS_CACHE=/app/cache/datasets

# Esponi porta per eventuale web interface futura
EXPOSE 8072

# Comando default
CMD ["python3", "server_green.py"]
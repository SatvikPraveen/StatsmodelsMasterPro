# ---- Base Python Image ----
FROM python:3.11-slim

# ---- Set Environment Variables ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- Set Work Directory ----
WORKDIR /app

# ---- Copy and Install Dependencies ----
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- Copy Project Files ----
COPY . .

# ---- Expose Ports for Streamlit and JupyterLab ----
EXPOSE 8501 8888

# ---- Entrypoint ----
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

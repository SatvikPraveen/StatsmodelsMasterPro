# 🐳 **Docker & Docker Compose – Ultimate Cheatsheet**

---

## 📦 Docker CLI Basics

| Action                              | Command                                   |
| ----------------------------------- | ----------------------------------------- |
| Check Docker version                | `docker --version`                        |
| List images                         | `docker images`                           |
| List running containers             | `docker ps`                               |
| List all containers (incl. stopped) | `docker ps -a`                            |
| Build image from Dockerfile         | `docker build -t <image_name> .`          |
| Run container from image            | `docker run -d -p 8501:8501 <image_name>` |
| Stop a container                    | `docker stop <container_id>`              |
| Remove a container                  | `docker rm <container_id>`                |
| Remove an image                     | `docker rmi <image_id>`                   |
| View container logs                 | `docker logs <container_id>`              |
| Access container shell              | `docker exec -it <container_id> bash`     |

---

## 🛠️ Docker Compose Essentials

| Action                    | Command                          |
| ------------------------- | -------------------------------- |
| Build & run in foreground | `docker compose up --build`      |
| Build & run in background | `docker compose up --build -d`   |
| Stop services             | `docker compose down`            |
| Rebuild specific service  | `docker compose build <service>` |
| List running services     | `docker compose ps`              |
| View logs                 | `docker compose logs -f`         |
| Stop a single service     | `docker compose stop <service>`  |
| Start a stopped service   | `docker compose start <service>` |

---

## 🗂️ Dockerfile Reminders

```Dockerfile
# Start from base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Expose port (for Streamlit or Jupyter)
EXPOSE 8501

# Run Streamlit or Jupyter
CMD ["streamlit", "run", "app.py"]
```

---

## 🧾 .dockerignore Template

```python
# Ignore local virtual environments
venv/
.env/
__pycache__/
*.pyc
*.pyo

# Ignore system files
.DS_Store
*.log

# Ignore data and export files
*.csv
*.xlsx
outputs/
exports/
```

---

## ⚙️ docker-compose.yml Skeleton

```yaml
services:
  app_service:
    build:
      context: .
    ports:
      - "8501:8501"
      - "8888:8888"
    volumes:
      - .:/app
    environment:
      - MODE=streamlit
    entrypoint: ["/bin/bash", "./entrypoint.sh"]
```

---

## 🔁 Useful One-Liners

| Description            | Command                          |
| ---------------------- | -------------------------------- |
| Stop all containers    | `docker stop $(docker ps -q)`    |
| Remove all containers  | `docker rm $(docker ps -aq)`     |
| Remove dangling images | `docker image prune`             |
| Remove all images      | `docker rmi $(docker images -q)` |
| Prune everything       | `docker system prune -a`         |

FROM python:3.11-slim

WORKDIR /app

# Install sigforge signal engine
COPY ../sigforge /sigforge
RUN pip install -e /sigforge

# Install sigwatch dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "run.py"]

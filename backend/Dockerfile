FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . /app/backend

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "backend/main.py"]



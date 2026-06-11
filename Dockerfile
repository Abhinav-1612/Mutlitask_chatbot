FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port (Hugging Face Spaces expects 7860 by default, Render expects 10000, we'll use an env var with a default)
ENV PORT=7860
EXPOSE $PORT

# Command to run the FastAPI backend
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]

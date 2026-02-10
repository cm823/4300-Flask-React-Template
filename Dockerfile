# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy frontend source
COPY frontend/ ./

# Build the React app
RUN npm run build

# Stage 2: Setup Python backend
FROM python:3.10-slim

ENV CONTAINER_HOME=/var/www

# Set working directory
WORKDIR $CONTAINER_HOME

# Copy requirements and install dependencies
COPY requirements.txt $CONTAINER_HOME/requirements.txt
RUN pip install --no-cache-dir -r $CONTAINER_HOME/requirements.txt

# Copy application files
COPY app.py models.py routes.py init.json $CONTAINER_HOME/

# Copy React build from frontend-build stage
COPY --from=frontend-build /app/frontend/dist $CONTAINER_HOME/frontend/dist

# Run the Flask application using gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]

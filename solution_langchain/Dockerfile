# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install ffmpeg and other necessary packages
# ffmpeg is crucial for audio extraction
# libsndfile1 is a dependency for some audio libraries
RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit application file into the container at /app
COPY app.py .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Define environment variables for Ollama host
# This allows the container to connect to Ollama running on the host machine.
# For Linux hosts, you might need to replace 'host.docker.internal' with your Docker bridge IP (e.g., 172.17.0.1)
ENV OLLAMA_HOST=host.docker.internal

# Run the Streamlit application when the container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Set the Python path to include /app
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PKL_FILE_PATH=/app/models/model.pkl

# Copy only the necessary files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5001

# Run the application
CMD ["python", "/app/src/app.py"]
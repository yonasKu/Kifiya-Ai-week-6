# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary dependencies (from requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

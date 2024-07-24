# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /home/tao/workspace/reef-imaging/reef_imaging/control/squid_microscope/squid-control

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Run the application
CMD ["python", "start_hypha_service.py"]

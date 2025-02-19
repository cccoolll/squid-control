FROM python:3.10.13-slim

# Create non-root user "squid-user"
RUN addgroup --system squid-user && adduser --system --ingroup squid-user squid-user

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app with proper ownership
COPY --chown=squid-user:squid-user . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install .

# Run the application as the non-root user "squid-user"
USER squid-user
CMD ["python", "start_hypha_service.py", "--simulation"]
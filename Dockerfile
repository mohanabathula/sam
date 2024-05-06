# Use ultralytics base image
FROM ultralytics/ultralytics:latest

# Set the working directory in the container
WORKDIR /app

# Update package lists and install necessary dependencies
RUN apt-get update && apt-get install -y python3-pip ffmpeg

# Copy the requirements file
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the SAM model and server code into the container
COPY sam_b.pt .
COPY server.py .

# Expose the port on which the server will listen
EXPOSE 8080

# Pull and Run with access to all GPUs
# t=ultralytics/ultralytics:latest && sudo docker pull $t && sudo docker run -it -p 8080:8080 $t

# Start the server
ENTRYPOINT ["python3", "server.py"]
# CMD ["python3", "server.py"]

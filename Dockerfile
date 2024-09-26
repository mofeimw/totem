FROM openvino/ubuntu20_runtime:latest

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set up OpenVINO environment
ENV PYTHONPATH=$PYTHONPATH:/opt/intel/openvino/python/python3.8
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/openvino/runtime/lib/intel64

# Use python3 explicitly in the CMD
CMD ["python3", "your_script.py"]

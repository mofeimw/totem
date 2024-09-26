FROM openvino/ubuntu20_dev:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install OpenShift AI client
RUN pip install openshift-client

# Set up OpenVINO environment
ENV PYTHONPATH=$PYTHONPATH:/opt/intel/openvino/python/python3.8
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/openvino/runtime/lib/intel64

CMD ["python", "backend/app.py"]

FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies (including libGL for OpenCV)
RUN yum update -y && yum install -y \
    gcc \
    gcc-c++ \
    make \
    libGL \
    libGLU \
    mesa-libGL \
    mesa-libGLU \
    && yum clean all

# Copy requirements and install Python dependencies
COPY src/requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# 🛠 Manually install paddleocr lightweight
RUN pip install "paddleocr>=2.6.1.3" --no-deps

# Copy function code
COPY src/lambda_function.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]

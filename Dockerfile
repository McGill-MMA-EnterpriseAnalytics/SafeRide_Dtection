FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies
RUN yum update -y && yum install -y \
    gcc \
    gcc-c++ \
    make \
    libGL \
    libGLU \
    mesa-libGL \
    mesa-libGLU \
    && yum clean all

# Install Python dependencies
COPY src/requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt


# Copy function code
COPY src/lambda_function.py ${LAMBDA_TASK_ROOT}

# Set handler
CMD [ "lambda_function.lambda_handler" ]

# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Install build tools and dependencies, including HDF5 libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    dbus \
    libhdf5-dev

# Pre-install Cython
RUN pip install Cython==0.29.21

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install the dependencies specified in the requirements file
RUN pip install -r requirements.txt

# Install specific versions of click and tornado
RUN pip install click==8.0.0
RUN pip install tornado==6.1
RUN pip install django
RUN pip install djangorestframework
RUN pip install opencv-python

RUN apt-get update && apt-get install -y \
libhdf5-dev \
libgl1-mesa-glx \
libglib2.0-0

# Copy the rest of the application code into the container
COPY . .

# Expose port 8000 for the Django server
EXPOSE 8000

# Set environment variable
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Run the Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

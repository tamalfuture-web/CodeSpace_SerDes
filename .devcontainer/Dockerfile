# Use a Microsoft-maintained Python image that's optimized for development containers.
FROM mcr.microsoft.com/devcontainers/python:3.11

# Copy the requirements file into the container
COPY requirements.txt /tmp/pip-tmp/
RUN pip install --no-cache-dir -r /tmp/pip-tmp/requirements.txt

# [Optional] If your models need other system-level packages, you can install them with apt-get.
# For example:
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-name>

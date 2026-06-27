FROM python:3.13
RUN apt-get update
WORKDIR /ds-mini-project
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
CMD ["bash"]
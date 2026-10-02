FROM arm64v8/python:3.12-slim

WORKDIR /workspace

RUN pip install --no-cache-dir \
    torch==2.14.1 \
    torchvision==0.29.1

COPY src /workspace/src
COPY docker/arm64-test/test_checkpoint.py /workspace/test_checkpoint.py

ENV PYTHONPATH=/workspace/src

CMD ["python", "test_checkpoint.py"]
# This is an example Dockerfile that builds a minimal container for running LK Agents
# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Install gcc, g++ and other build dependencies.
RUN apt-get update && \
    apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    # As long as there are pip dependencies in requirements.txt file declared as git repositories the below package is needed
    git \
    && rm -rf /var/lib/apt/lists/*

USER appuser

RUN mkdir -p /home/appuser/.cache
RUN chown -R appuser /home/appuser/.cache

WORKDIR /home/appuser

# This allows invalidating the image building cache from this point to
# force the re-installation of depedencies and re-copying the source code
# "docker build -t openvidu/agent-AGENT_NAME:main --build-arg CACHEBUST=$(date +%s) ."
ARG CACHEBUST=1

COPY requirements.txt .
RUN python -m pip install --user --no-cache-dir -r requirements.txt

COPY ./*.py .

# ensure that any dependent models are downloaded at build-time
RUN python main.py download-files

# Run the application.
ENTRYPOINT ["python", "main.py"]
CMD ["start"]
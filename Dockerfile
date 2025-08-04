# This follows:
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
# https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
# With some input from:
# https://github.com/pipecat-ai/pipecat/blob/main/examples/telnyx-chatbot/Dockerfile
# https://depot.dev/docs/container-builds/how-to-guides/optimal-dockerfiles/python-uv-dockerfile
#

# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install dependencies (?)
# RUN --mount=type=cache,target=/root/.cache/uv \
#     --mount=type=bind,source=uv.lock,target=uv.lock \
#     --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
#     uv sync --locked --no-install-project
# RUN uv sync --locked --no-install-project --no-cache-dir

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, asserting the lockfile is up to date
RUN uv sync --locked --no-dev --no-cache-dir

# Copy .env file if it exists (for local development)
# This will be ignored in production where env vars are passed directly
# commented out for now to test zero downtime deploy
# COPY .env* ./

# Expose the desired port (matching main.py port 4000)
EXPOSE 4000

# Run the application
# CMD ["uv", "run", "--python 3.12", "main.py"]
CMD ["uv", "run", "main.py"]

FROM python:3.12.3-slim-bookworm

ENV POETRY_VERSION 1.8.0

# Install system dependencies
RUN pip install "poetry==$POETRY_VERSION"

# Install Node.js (LTS) for build smoke tests (npm install + npm run build).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy only requirements to cache them in docker layer
COPY poetry.lock pyproject.toml /app/

# Disable the creation of virtual environments
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install

# Install Chromium and Linux libraries needed by Playwright screenshot previews.
RUN playwright install --with-deps chromium

# Copy the current directory contents into the container at /app
COPY ./ /app/

EXPOSE 7001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7001"]

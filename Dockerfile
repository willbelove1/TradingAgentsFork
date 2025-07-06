# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
# Ensures Python output is sent straight to terminal without buffering
ENV PYTHONUNBUFFERED 1
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by some Python packages
# (e.g., for certain C extensions). This is a common set, might need adjustment.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    # Add other system dependencies if required by your Python packages
    && rm -rf /var/lib/apt/lists/*

# Install pipenv if you were using it (based on uv.lock, maybe uv or just pip)
# For now, assuming requirements.txt is the primary source of dependencies.
# If uv is preferred and can generate a requirements.txt or be used directly:
# RUN pip install uv

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# If using uv: RUN uv pip install --system --no-cache -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make the CLI executable (if it's a script that needs it, though Typer apps usually don't)
# RUN chmod +x /app/cli/main.py

# Expose any ports the application might use (if it's a web server, Chainlit, etc.)
# For a CLI tool, this is usually not needed unless it starts a server.
# Chainlit default port is 8000. If chainlit is used:
# EXPOSE 8000

# Define the entrypoint for the container.
# This makes the container executable and runs the CLI.
# Users can append arguments like "analyze --profile fast" when running the container.
ENTRYPOINT ["python", "-m", "cli.main"]

# Default command (can be overridden by user)
# For example, to show help by default:
# CMD ["--help"]
# Or to run analyze by default (though usually users will specify this)
# CMD ["analyze"]
# If no default command is desired, ENTRYPOINT is sufficient.
# Let's make it run analyze by default, users can override.
CMD ["analyze"]

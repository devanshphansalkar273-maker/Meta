# Use standard Python 3.10 slim variant for a minimal, optimized footprint
FROM python:3.10-slim

# Set strict production environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Hugging Face spaces optimization: reduce memory overhead footprint
    MALLOC_ARENA_MAX=2

# Set Hugging Face Spaces specific user permissions
# HF Spaces run as a random UID or user `1000`. Safe practice to run as a non-root standard user.
RUN useradd -m -u 1000 user
USER user

# Set working directory to user home
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Only copy requirements.txt first to leverage Docker layer caching
COPY --chown=user:user requirements.txt .

# Install dependencies, ensuring no cache is left behind to streamline image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy remaining project files with correct user permissions
COPY --chown=user:user . .

# Run inference.py directly
CMD ["python", "inference.py"]

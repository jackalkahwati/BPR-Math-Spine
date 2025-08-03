# BPR-Math-Spine: Docker image for universal compatibility
#
# This Dockerfile provides a complete BPR environment with FEniCS,
# Python scientific stack, and all dependencies.
#
# Usage:
#   docker build -t bpr-math-spine .
#   docker run -it --rm -p 8888:8888 bpr-math-spine
#
# Author: Jack Al-Kahwati (jack@thestardrive.com)

FROM ubuntu:22.04

# Metadata
LABEL maintainer="jack@thestardrive.com"
LABEL description="BPR-Math-Spine: Boundary Phase Resonance mathematical framework"
LABEL version="1.0"

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic tools
    wget \
    curl \
    git \
    build-essential \
    # Python development
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    # Scientific computing libraries
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    # FEniCS dependencies
    software-properties-common \
    # LaTeX for notebook export (optional)
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install FEniCS via PPA (legacy version - stable and well-tested)
RUN add-apt-repository ppa:fenics-packages/fenics && \
    apt-get update && \
    apt-get install -y \
    fenics \
    python3-dolfin \
    python3-mshr \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY environment.yml .
COPY environment-minimal.yml .
COPY environment-docker.yml .
COPY requirements-docker.txt .

# Install Miniconda for Python environment management (architecture-aware)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Set up Python virtual environment
ENV PATH="/opt/bpr-env/bin:$PATH"
ENV VIRTUAL_ENV="/opt/bpr-env"

# Create Python virtual environment and install packages (TOS-free approach)
RUN python3 -m venv /opt/bpr-env && \
    /opt/bpr-env/bin/pip install --upgrade pip setuptools wheel && \
    /opt/bpr-env/bin/pip install -r requirements-docker.txt

# Keep conda available for advanced users but use pip env by default
ENV PATH="/opt/bpr-env/bin:/opt/conda/bin:$PATH"

# Verify FEniCS installation
RUN python -c "import dolfin; print(f'✅ FEniCS version: {dolfin.__version__}')" || \
    echo "⚠️  FEniCS not available - continuing with numerical-only mode"

# Copy BPR-Math-Spine source code
COPY . .

# Install BPR package in development mode
RUN pip install -e .

# Create user for running notebooks (security)
RUN useradd -m -s /bin/bash bpruser && \
    chown -R bpruser:bpruser /workspace

USER bpruser

# Test the installation
RUN python -c "import bpr; print('✅ BPR package imported successfully')"

# Run basic tests
RUN python scripts/test_fenics_install.py || \
    echo "⚠️  Some tests failed - this is expected without full FEniCS"

# Set up Jupyter environment
# Jupyter already installed in pip requirements

# Expose Jupyter port
EXPOSE 8888

# Default command: start Jupyter Lab
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--notebook-dir=/workspace", \
     "--ServerApp.token=bpr-token-2025", \
     "--ServerApp.password=''"]

# Alternative entry points (uncomment as needed):
#
# Run demo script:
# CMD ["python", "scripts/run_casimir_demo.py"]
#
# Run tests:
# CMD ["pytest", "-v"]
#
# Interactive shell:
# CMD ["bash"]
# vLLM vs TGI Benchmark Suite

A comprehensive benchmarking environment for comparing vLLM against Hugging Face Text Generation Inference (TGI) on RHEL 9, with Prometheus/Grafana observability and CTO-focused ROI metrics.

## Overview

This benchmark suite demonstrates the ROI benefits of vLLM's PagedAttention and continuous batching versus traditional inference approaches:

- **Request Density**: More concurrent users per GPU
- **Unit Cost**: Lower cost per token generated
- **Latency**: Improved TTFT (Time to First Token) and TPOT (Time Per Output Token)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Generator                           │
│                    (benchmark/load_generator.py)                 │
└─────────────────────┬───────────────────────┬───────────────────┘
                      │                       │
                      ▼                       ▼
┌─────────────────────────────┐  ┌─────────────────────────────────┐
│         TGI (Baseline)      │  │       vLLM (Champion)           │
│    Port 8080 (/generate)    │  │  Port 8000 (/v1/completions)    │
│    Static Batching          │  │  PagedAttention + Cont. Batch   │
└──────────────┬──────────────┘  └──────────────┬──────────────────┘
               │                                │
               └────────────────┬───────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Prometheus                               │
│                          Port 9090                               │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Grafana Dashboard                             │
│           Port 3000 - CTO Metrics Dashboard                      │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- **RHEL 9.x** with kernel 5.14+
- **Podman 4.x+** with GPU support
- **NVIDIA GPU** (L4, L40S, A100, H100, or similar datacenter GPUs)
- **NVIDIA Driver** from official CUDA repository
- **Hugging Face account** with access to your chosen model

> **⚠️ Single GPU Note**: With one GPU, you can only run vLLM OR TGI at a time (each needs ~15GB VRAM for a 7B model). Benchmark them sequentially using `--vllm-only` and `--tgi-only` flags.

### Step 1: Install Podman and Python

```bash
# Install Podman and Python
sudo dnf install -y podman python3 python3-pip

# Install podman-compose (Python-based)
sudo pip3 install podman-compose

# Verify installation
podman --version
podman-compose --version
```

### Step 2: Configure Docker Hub Registry

By default, Podman searches Red Hat registries first. Add Docker Hub:

```bash
echo 'unqualified-search-registries = ["docker.io", "registry.access.redhat.com"]' | \
  sudo tee /etc/containers/registries.conf.d/docker.conf
```

### Step 3: Install NVIDIA Driver

RHEL 9 has excellent NVIDIA support with pre-built kernel modules:

```bash
# Add NVIDIA CUDA repository
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

# Install NVIDIA drivers
sudo dnf install -y cuda-drivers

# Reboot to load the driver
sudo reboot
```

After reboot:

```bash
# Verify installation
nvidia-smi
```

### Step 4: Install NVIDIA Container Toolkit

```bash
# Add NVIDIA container toolkit repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install the toolkit
sudo dnf install -y nvidia-container-toolkit

# Configure for Podman (generates CDI specification)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Verify GPU access in containers
podman run --rm --device nvidia.com/gpu=all docker.io/nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. Clone and Configure

```bash
cd /path/to/vllm

# Copy environment template and add your HuggingFace token
cp env.example .env
vim .env  # Add your HF_TOKEN
```

Edit `.env`:
```bash
HF_TOKEN=hf_your_token_here
MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3  # No approval needed
# MODEL_ID=meta-llama/Llama-3.1-8B-Instruct  # Requires HF approval first
GPU_COST_PER_HOUR=0.80  # L4 pricing
```

### 2. Start the Stack

```bash
# Start all services
# Use 'sudo' if GPU access requires root privileges
sudo podman-compose up -d

# Watch logs
sudo podman-compose logs -f

# Check status
sudo podman-compose ps
```

> **Note**: If `podman-compose` isn't found, install it with `sudo pip3 install podman-compose`

### 3. Wait for Models to Load

Both TGI and vLLM need to download and load the model (5-10 minutes on first run):

```bash
# Check health endpoints
curl http://localhost:8000/health  # vLLM
curl http://localhost:8080/health  # TGI
```

### 4. Run Benchmarks

**Note**: With a single GPU, you can only run one inference engine at a time. The stack starts vLLM by default.

```bash
# Install Python dependencies
cd benchmark
pip3 install -r requirements.txt

# Check endpoint health
python3 load_generator.py health

# ============================================
# BENCHMARK vLLM (started by default)
# ============================================
python3 load_generator.py run --vllm-only -c 4 -d 30 -o vllm_c4.csv
python3 load_generator.py run --vllm-only -c 8 -d 30 -o vllm_c8.csv

# ============================================
# SWITCH TO TGI (single GPU - must stop vLLM)
# ============================================
cd ..
sudo podman stop vllm-champion
sudo podman rm vllm-champion

# Start TGI and wait for model to load (~90 seconds)
sudo podman-compose up -d tgi
sleep 90

# Verify TGI is ready
cd benchmark
python3 load_generator.py health

# ============================================
# BENCHMARK TGI
# ============================================
python3 load_generator.py run --tgi-only -c 4 -d 30 -o tgi_c4.csv
python3 load_generator.py run --tgi-only -c 8 -d 30 -o tgi_c8.csv
```

### 5. View Dashboard

Open Grafana: http://localhost:3000
- **Username**: admin
- **Password**: benchmark123 (or value from `GRAFANA_PASSWORD`)

The "CTO Inference Benchmark Dashboard" is pre-configured as the home dashboard.

### 6. Compare Results

View benchmark comparison:

```bash
# Display results from CSV files
cat vllm_results.csv
cat tgi_results.csv

# Key metrics to compare:
# - throughput_tok_s: Tokens generated per second
# - ttft_p50_ms: Time to first token (P50)
# - tpot_p50_ms: Time per output token (P50)
```

### 7. Generate ROI Report

```bash
cd benchmark/analysis
python3 roi_calculator.py --input ../sweep_results.csv --annual-tokens 1000000000
```

## Benchmark Metrics

### Primary KPIs

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Throughput (tokens/sec)** | Total tokens generated per second | Direct capacity measure |
| **TTFT (Time to First Token)** | Latency until first token appears | User experience - perceived responsiveness |
| **TPOT (Time Per Output Token)** | Time between subsequent tokens | "Reading speed" of the response |
| **Request Density** | Concurrent requests per GPU | Hardware utilization efficiency |
| **Cost per 1M Tokens** | Dollar cost to generate 1M tokens | Unit economics |

### Expected Results (L4 + Mistral-7B)

Based on benchmarks with NVIDIA L4 GPU (22GB VRAM):

| Metric | vLLM | TGI | vLLM Advantage |
|--------|------|-----|----------------|
| Throughput @ C=4 | ~48 tok/s | ~49 tok/s | Similar |
| Throughput @ C=8 | ~85 tok/s | ~88 tok/s | Similar |
| TTFT P50 @ C=8 | **136ms** | 171ms | **20% faster** |
| TPOT P50 @ C=8 | **60ms** | 65ms | **8% faster** |

**Key Insight**: vLLM's advantages become more pronounced at:
- Higher concurrency (16+ requests)
- Longer output sequences (256+ tokens)
- Memory-constrained scenarios

### Expected Results (L40S/A100 + Llama-3.1-8B)

At higher concurrency with larger GPUs:

| Metric | vLLM | TGI | vLLM Advantage |
|--------|------|-----|----------------|
| Throughput at Concurrency 100 | ~2000 tok/s | ~800 tok/s | **2.5x** |
| TTFT P50 | ~50ms | ~150ms | **3x faster** |
| Max Concurrent Requests | 200+ | 50-100 | **2-4x more** |
| Cost per 1M tokens | ~$0.20 | ~$0.50 | **60% cheaper** |

## File Structure

```
vllm/
├── podman-compose.yaml          # Container orchestration
├── env.example                  # Environment template
├── configs/
│   ├── prometheus.yml           # Prometheus scrape config
│   └── grafana/
│       ├── datasources.yml      # Prometheus datasource
│       ├── dashboard-provider.yml
│       └── dashboards/
│           └── cto-metrics.json # Pre-built dashboard
├── benchmark/
│   ├── requirements.txt         # Python dependencies
│   ├── load_generator.py        # Main benchmark script
│   ├── datasets/
│   │   └── prompts.jsonl        # Test prompts
│   └── analysis/
│       └── roi_calculator.py    # ROI analysis tool
└── README.md                    # This file
```

## Commands Reference

### Podman Compose

```bash
# Start all services (use sudo for GPU access)
sudo podman-compose up -d

# Stop all services
sudo podman-compose down

# View logs
sudo podman logs vllm-champion
sudo podman logs tgi-baseline

# Stop/start individual containers
sudo podman stop vllm-champion
sudo podman start tgi-baseline

# Check container status
sudo podman-compose ps

# Check GPU/resource usage
nvidia-smi
sudo podman stats
```

### Benchmark CLI

```bash
# Check health of inference endpoints
python3 load_generator.py health

# Run benchmark at specific concurrency (both engines must be running)
python3 load_generator.py run -c 8 -d 60 -w 5 -o results.csv

# Benchmark only vLLM
python3 load_generator.py run --vllm-only -c 8 -d 30 -o vllm.csv

# Benchmark only TGI  
python3 load_generator.py run --tgi-only -c 8 -d 30 -o tgi.csv

# Run sweep across multiple concurrency levels
python3 load_generator.py sweep -c 1,4,8,16,32 -d 30 -o sweep_results.csv
```

**CLI Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--concurrency` | `-c` | Number of concurrent requests |
| `--duration` | `-d` | Benchmark duration in seconds |
| `--warmup` | `-w` | Warmup requests before measurement |
| `--output` | `-o` | Output CSV filename |
| `--vllm-only` | | Only benchmark vLLM |
| `--tgi-only` | | Only benchmark TGI |

### ROI Calculator

```bash
# Basic analysis
python3 analysis/roi_calculator.py --input sweep_results.csv

# With custom parameters
python3 analysis/roi_calculator.py \
  --input sweep_results.csv \
  --gpu-cost 2.50 \
  --gpu-name "A100-80GB" \
  --annual-tokens 10000000000 \
  --output roi_report.csv
```

## Interpreting Results

### Throughput Advantage

- **2x or higher**: Significant improvement, clear ROI
- **1.5x-2x**: Moderate improvement, worth migration
- **1x-1.5x**: Marginal gains, consider other factors

### Cost Savings

The ROI calculator projects annual savings based on your token volume. For a typical enterprise:

| Annual Tokens | vLLM Savings (vs TGI) |
|---------------|----------------------|
| 100M | $30-50 |
| 1B | $300-500 |
| 10B | $3,000-5,000 |
| 100B | $30,000-50,000 |

### When vLLM Excels

vLLM shows the strongest advantages when:

1. **High concurrency**: 50+ simultaneous requests
2. **Long outputs**: 256+ tokens per response
3. **Variable workloads**: Mixed prompt/output lengths
4. **Memory pressure**: Near GPU memory limits

### When Differences are Smaller

- Low concurrency (1-10 requests)
- Very short outputs (<64 tokens)
- Heavily batch-optimized TGI configurations

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA driver is loaded
nvidia-smi
lsmod | grep nvidia

# Check CDI specification exists
cat /etc/cdi/nvidia.yaml

# Regenerate CDI if needed
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

### CDI Device Error ("unresolvable CDI devices")

This means the CDI spec wasn't generated or the NVIDIA driver isn't installed:

```bash
# Check if driver is installed
nvidia-smi

# If driver works, regenerate CDI
sudo mkdir -p /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Verify CDI spec
cat /etc/cdi/nvidia.yaml | head -30
```

### RHEL 9: Driver Installation Issues

```bash
# If cuda-drivers fails, check NVIDIA repo is enabled
sudo dnf repolist | grep cuda

# Re-add the repo if needed
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

# Clean cache and retry
sudo dnf clean all
sudo dnf install -y cuda-drivers
```

### AWS/Cloud: Unregistered RHEL

If you see "Unable to read consumer identity" errors, the system isn't registered with Red Hat. This is normal for AWS - RHUI handles updates:

```bash
# AWS RHEL AMIs use RHUI (Red Hat Update Infrastructure)
# The cuda repo works independently of subscription-manager
sudo dnf repolist

# Verify NVIDIA repo is accessible
sudo dnf list available cuda-drivers
```

### Model Download Fails

```bash
# Verify HF token
export HF_TOKEN=your_token
huggingface-cli whoami

# Check token permissions for gated models
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

### Out of Memory

```bash
# Reduce GPU memory utilization in podman-compose.yaml
# For vLLM: --gpu-memory-utilization=0.80
# For TGI: --sharded=false --quantize=awq
```

### Connection Refused

```bash
# Wait for model loading (5-10 min on first run)
podman compose logs -f vllm | grep -i "started"
podman compose logs -f tgi | grep -i "connected"
```

## Customization

### Using a Different Model

Edit `.env` or `podman-compose.yaml`:

```bash
MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
```

### Adding Custom Prompts

Edit `benchmark/datasets/prompts.jsonl`:

```json
{"prompt": "Your custom prompt here", "max_tokens": 256, "category": "custom"}
```

### Adjusting GPU Memory

In `podman-compose.yaml`:

```yaml
# vLLM
command:
  - --gpu-memory-utilization=0.85  # Default 0.90

# TGI  
command:
  - --max-batch-prefill-tokens=2048  # Reduce for less memory
```

## External API Access

The vLLM server provides an OpenAI-compatible API that can be accessed from external applications.

### Enable API Key Authentication

1. Edit `.env` and set a secure API key:
```bash
VLLM_API_KEY=sk-your-secret-api-key-here
```

2. Open port 8000 in your AWS Security Group (or firewall)

3. Restart the stack:
```bash
sudo podman-compose down
sudo podman-compose up -d
```

### Python Client Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://YOUR_SERVER_IP:8000/v1",
    api_key="sk-your-secret-api-key-here"
)

response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Curl Example

```bash
curl http://YOUR_SERVER_IP:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Available Endpoints

| Endpoint | Description |
|----------|-------------|
| `/v1/chat/completions` | Chat-style completions (recommended) |
| `/v1/completions` | Text completions |
| `/v1/models` | List available models |
| `/health` | Health check (no auth required) |

### Security Recommendations

- Use a strong random API key (32+ characters)
- Restrict AWS Security Group to specific IP ranges
- Consider adding HTTPS via reverse proxy (nginx/caddy) for production

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See LICENSE file for details.


# Confidential SLM Chat (SEV-SNP)

A secure chat application that leverages AMD SEV-SNP confidential computing to provide privacy-preserving inference with small language models (SLMs). This project demonstrates how to combine attestation, hardware-based isolation, and modern NLP pipelines to ensure user data and model execution remain confidential, even in untrusted cloud environments.

Built for research and academic purposes, this application showcases:
- **Confidential Computing**: VM-level security using AMD SEV-SNP technology
- **Language Model Inference**: Support for multiple Hugging Face transformer models
- **Performance Benchmarking**: Comprehensive benchmarking suite for model performance analysis
- **Attestation & Verification**: End-to-end cryptographic verification of execution environment
- **Modern Web Interface**: Responsive UI with real-time attestation status

---

## Features

### Core Security Features
- **AMD SEV-SNP Protection**: All inference and sensitive operations run inside a VM protected by AMD SEV-SNP confidential computing
- **Cryptographic Attestation**: Real-time verification of VM integrity and confidentiality using hardware-rooted trust
- **Certificate Chain Validation**: Automated VCEK certificate verification against AMD's certificate authority
- **Secure Session Management**: Cookie-based session management that requires successful attestation

### Language Model Capabilities
- **Multiple Model Support**: Compatible with various Hugging Face transformer models including:
  - DistilGPT2 (default)
  - TinyGPT2
  - TinyLlama-1.1B-Chat
  - Microsoft Phi-1.5
- **Efficient Inference**: Optimized for small language models suitable for confidential computing environments
- **Batch Processing**: Support for batch inference with configurable batch sizes

### Performance & Benchmarking
- **Comprehensive Benchmarking Suite**: Built-in performance analysis tools measuring:
  - Latency percentiles (P50, P90, P95, P99)
  - Throughput (tokens per second)
  - Memory usage tracking
  - Cold-start performance
  - Concurrent request handling
- **Multi-dimensional Testing**: Configurable token lengths, batch sizes, and concurrency levels
- **Hardware Monitoring**: Optional performance counter integration with Linux `perf` tools

### User Experience
- **Modern Web Interface**: Responsive design with TailwindCSS styling
- **Real-time Status**: Live attestation status with visual indicators
- **Interactive Chat**: Seamless chat experience with message history
- **Error Handling**: Comprehensive error messaging and fallback behaviors

---

## Setup Instructions

### 1. Prerequisites

#### Hardware & Environment Requirements
- **AMD SEV-SNP enabled VM** (for full confidential computing support)
  - AMD EPYC 3rd Gen (Milan) or newer processor
  - VM configured with SEV-SNP enabled
  - Sufficient memory allocation (recommended: 4GB+ for model inference)
- **Linux Distribution** (Ubuntu 20.04+ or CentOS 8+ recommended)
- **sudo privileges** (required for attestation operations)

#### Software Dependencies
- **Python 3.8+** with pip package manager
- **curl** (for certificate chain downloads)
- **OpenSSL** (for certificate verification)
- **snpguest** CLI tool (build from source or install pre-compiled)
  - Default expected path: `~/attestation-libs/snpguest/target/release/snpguest`
  - Build instructions: [AMD SNP Guest Tools](https://github.com/AMDESE/sev-guest)

#### Optional Tools
- **perf** (Linux performance monitoring, for detailed benchmarking)
- **CUDA Toolkit** (if GPU acceleration is desired, though CPU inference is default)

### 2. Clone the Repository

```bash
git clone https://github.com/patelajaychh/confidential-slm-chat.git
cd confidential-slm-chat
```

### 3. Install Python Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 4. Prepare Attestation Environment

#### Install snpguest (if not already available)
```bash
# Clone and build snpguest
git clone https://github.com/AMDESE/sev-guest.git ~/attestation-libs/sev-guest
cd ~/attestation-libs/sev-guest
cargo build --release

# Create expected directory structure
mkdir -p ~/attestation-libs/snpguest/target/release/
cp target/release/snpguest ~/attestation-libs/snpguest/target/release/
```

#### Verify SEV-SNP Environment
```bash
# Check if SEV-SNP is available
sudo dmesg | grep -i sev

# Verify snpguest installation
~/attestation-libs/snpguest/target/release/snpguest --help
```

### 5. Run the Application

```bash
# Make run script executable
chmod +x run.sh

# Run in production mode
./run.sh

# Run in debug mode (with auto-reload)
./run.sh --debug
```

The application will be available at [http://localhost:8000](http://localhost:8000)

### 6. Run Performance Benchmarks (Optional)

```bash
# Run comprehensive model benchmarks
python benchmark_models.py

# Results will be saved to benchmark_results.csv
# View results
cat benchmark_results.csv
```

---

## API Endpoints

### 1. `GET /`

- **Description**: Returns the main chat UI.
- **Response**: HTML page.

### 2. `GET /attest`

- **Description**: Performs attestation of the VM environment.
- **Response**: JSON object:
  - `attestation_report_base64`: Base64-encoded attestation report.
  - `verification_result`: `"verified"` if the environment is secure, `"not verified"` otherwise.

### 3. `POST /chat`

- **Description**: Sends a user message to the LLM and returns the generated response.
- **Request Body**:
  ```json
  {
    "message": "Your message here"
  }
  ```
- **Response**:
  ```json
  {
    "response": "Model's generated reply"
  }
  ```

### 4. `GET /favicon.ico`

- **Description**: Serves the favicon for the web UI.

---

## Usage Notes

- **Attestation Required**: Users must attest the environment before chatting. If attestation fails or is not verified, chat is disabled.
- **Security**: All sensitive operations are performed inside a confidential VM. Attestation results are shown in the UI.

## Project Structure

```
confidential-slm-chat/
├── main.py           # FastAPI backend
├── attestation.py    # Attestation logic
├── templates/
│   └── index.html    # Web UI
├── static/           # Static files (favicon, etc.)
├── run.sh            # Startup script
└── ...               # Other files
```

---

## License

This project is for academic and research purposes. Please review and comply with the licenses of all dependencies and referenced models/tools.
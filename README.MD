# Confidential SLM Chat (SEV-SNP)

A secure chat application that leverages AMD SEV-SNP confidential computing to provide privacy-preserving inference with a large language model (LLM). This project demonstrates how to combine attestation, hardware-based isolation, and modern NLP pipelines to ensure user data and model execution remain confidential, even in untrusted cloud environments.

---

## Features

- **Confidential Computing**: All inference and sensitive operations run inside a VM protected by AMD SEV-SNP.
- **Attestation**: Users can verify the integrity and confidentiality of the environment before chatting.
- **Modern LLM**: Uses HuggingFace's `pipeline` for text generation (default: DistilGPT2, can be switched to TinyLlama).

---

## Setup Instructions

### 1. Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)
- **Node.js** (optional, for advanced frontend development)
- **AMD SEV-SNP enabled VM** (for full confidential computing support)
- **snpguest** CLI tool installed on VM.

### 2. Clone the Repository

```bash
git clone https://github.com/patelajaychh/confidential-slm-chat.git
cd confidential-slm-chat
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Attestation Tools

- Ensure `snpguest` CLI tool is available at `~/attestation-libs/snpguest/target/release/snpguest`
- The backend expects this path by default. Adjust in `attestation.py` if needed.

### 5. Run the Application

```bash
# Run in production mode
./run.sh

# Run in debug mode
./run.sh --debug
```

The app will be available at [http://localhost:8000](http://localhost:8000).

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
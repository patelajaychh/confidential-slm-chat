import os 
import subprocess

class Attestation:
    def __init__(self, snpguest_path=None):
        self.snpguest_path = snpguest_path or os.path.join(os.path.expanduser("~"), "attestation-libs/snpguest/target/release", "snpguest")
        
        self.artifact_dir = os.path.join(os.path.expanduser("~"), "attestation-libs/snpguest/tmp/")
        if not os.path.exists(self.artifact_dir):
            os.makedirs(self.artifact_dir)
        
        self.device_certs_dir = os.path.join(self.artifact_dir, "device_certs")
        if not os.path.exists(self.device_certs_dir):
            os.makedirs(self.device_certs_dir)

        self.vcek_cert_path = os.path.join(self.device_certs_dir, "vcek.pem")

    def generate_report(self):
        report_path = os.path.join(self.artifact_dir, "attestation_report.bin")
        nonce_file_path = os.path.join(self.artifact_dir, "nonce.txt")

        if not os.path.exists(self.snpguest_path):
            raise FileNotFoundError(f"snpguest utility not found at {self.snpguest_path}")

        if os.path.exists(report_path):
            print(f"Removing existing report file: {report_path}")
            os.remove(report_path)
        if os.path.exists(nonce_file_path):
            print(f"Removing existing nonce file: {nonce_file_path}")
            os.remove(nonce_file_path)

        # Create an empty nonce file
        print(f"Creating empty nonce file.")
        with open(nonce_file_path, "wb") as f:
            pass

        # Generate guest report using snpguest
        result = subprocess.run([
            "sudo", self.snpguest_path,
            "report", report_path,
            nonce_file_path, 
            "--random"
        ], capture_output=True, check=True)

        if result.returncode != 0:
            print(result.stderr.strip())
            raise RuntimeError("Failed to generate attestation report")
        
        print(f"Attestation report generated at {report_path}")
        return report_path
        
    def request_devie_certificates(self):
        print(f"Requesting device certificates and storing as PEM files in {self.device_certs_dir}")
        result = subprocess.run([
            "sudo", self.snpguest_path,
            "certificates",
            "PEM",
            self.device_certs_dir
        ], capture_output=True, check=True)
        if result.returncode != 0:
            print(result.stderr.strip())
            raise RuntimeError("Failed to request certificates")

    def download_cert_chain(self, cert_chain_path):
        url = "https://kdsintf.amd.com/vcek/v1/Milan/cert_chain"
        print(f"Downloading AMD Root certificate chain to {cert_chain_path}")
        result = subprocess.run([
            "curl",
            "--proto", "=https",
            "--tlsv1.2",
            "-sSf",
            url,
            "-o", cert_chain_path
        ], check=True)
        if result.returncode != 0:
            raise RuntimeError("Failed to download cert_chain.pem")

    def verify_vcek_certificate(self, cert_chain_path):
        self.request_devie_certificates()
        print(f"Verifying VCEK certificate {self.vcek_cert_path} with chain {cert_chain_path}")
        result = subprocess.run([
            "openssl", "verify",
            "--CAfile", cert_chain_path,
            self.vcek_cert_path
        ], capture_output=True, text=True)

        print(result.stderr.strip())
        if result.returncode != 0 or "OK" not in result.stdout:
            raise RuntimeError(f"VCEK certificate verification failed: {result.stdout} {result.stderr}")
        print(result.stdout.strip())

    def verify_report(self):
        report_path = os.path.join(self.artifact_dir, "attestation_report.bin")
    
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Attestation report not found at {report_path}")
        if not os.path.exists(self.device_certs_dir):
            raise FileNotFoundError(f"Device certificates directory not found at {self.device_certs_dir}")
        
        # Download cert chain and verify VCEK certificate
        cert_chain_path = os.path.join(self.artifact_dir, "cert_chain.pem")
        self.download_cert_chain(cert_chain_path)
        self.verify_vcek_certificate(
            cert_chain_path=cert_chain_path)

        # Verify attestation report
        print(f"Verifying attestation report {report_path}")
        result = subprocess.run([
            self.snpguest_path,
            "verify",
            "attestation",
            self.device_certs_dir,
            report_path
        ], capture_output=True, text=True)
        
        output = result.stdout.strip() + "\n" + result.stderr.strip()
        print(output)
        if result.returncode == 0 and all(
            phrase in output for phrase in [
                "Reported TCB Boot Loader from certificate matches the attestation report.",
                "Reported TCB TEE from certificate matches the attestation report.",
                "Reported TCB SNP from certificate matches the attestation report.",
                "Reported TCB Microcode from certificate matches the attestation report.",
                "VEK signed the Attestation Report!"
            ]
        ):
            return "verified"
        elif result.returncode == 0:
            return "not verified"
        else:
            raise RuntimeError(f"Attestation report verification failed: {result.stdout} {result.stderr}")
        
    
"""
TLS Certificate Generator and SSL Configuration Utility.
Generates self-signed certificates for local HTTPS testing and mutual TLS drone connections.
"""

import sys
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("tls_config")


def generate_self_signed_cert(cert_dir: Path = None, common_name: str = "localhost"):
    """Generates a self-signed X.509 certificate and RSA private key."""
    if cert_dir is None:
        cert_dir = settings.BASE_DIR / "security" / "certs"
    cert_dir.mkdir(parents=True, exist_ok=True)

    key_path = cert_dir / "server.key"
    cert_path = cert_dir / "server.crt"

    # Generate Private Key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Write Private Key
    with open(key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Build Certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SkyGuard UAV Network"),
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
    ])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(common_name), x509.DNSName("127.0.0.1")]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    # Write Certificate
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    logger.info(f"Generated TLS Certificate: {cert_path}")
    logger.info(f"Generated TLS Private Key: {key_path}")
    return cert_path, key_path


if __name__ == "__main__":
    generate_self_signed_cert()

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime
import os

def generate_self_signed_cert():
    # Generate key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Generate certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"CN"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Shanghai"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"Shanghai"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"KimiLocal"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"192.168.230.2"),
    ])

    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        # Valid for 10 years
        datetime.datetime.utcnow() + datetime.timedelta(days=3650)
    ).add_extension(
        x509.SubjectAlternativeName([x509.IPAddress(
            # Add IP SAN to avoid some browser warnings (though self-signed will always warn)
            import_ip_address("192.168.230.2")
        )]),
        critical=False,
    ).sign(key, hashes.SHA256())

    # Write key
    with open("key.pem", "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    # Write cert
    with open("cert.pem", "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print("Certificate (cert.pem) and Key (key.pem) generated successfully.")

def import_ip_address(ip_str):
    import ipaddress
    return ipaddress.ip_address(ip_str)

if __name__ == "__main__":
    try:
        # Install required package if missing
        # But usually we are in venv
        generate_self_signed_cert()
    except ImportError:
        print("Installing cryptography...")
        os.system("pip install cryptography")
        generate_self_signed_cert()


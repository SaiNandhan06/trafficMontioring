"""
Security & Repository Hygiene Regression Tests.
Verifies .gitignore rules, secret safety, dataset path portability, and production secret enforcement.
"""

from pathlib import Path
import pytest
import yaml
from config.settings import Settings, BASE_DIR


def test_gitignore_exists_and_protects_sensitive_files():
    """Verifies root .gitignore exists and specifies rules for secrets, DBs, logs, and caches."""
    gitignore_path = BASE_DIR / ".gitignore"
    assert gitignore_path.exists(), "Root .gitignore must exist"

    content = gitignore_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    # Critical patterns that must be ignored
    required_patterns = [
        ".env",
        "*.db",
        "logs/",
        "ipfs/mock_store/",
        ".cache/",
        "security/.vault.enc"
    ]

    for pattern in required_patterns:
        assert any(pattern in line for line in lines), f"Missing pattern '{pattern}' in .gitignore"

    # Ensure source code is not accidentally ignored
    forbidden_patterns = ["config/", "edge/", "blockchain/", "dashboard/", "ipfs/", "src/"]
    for pattern in forbidden_patterns:
        assert pattern not in lines, f"Source directory '{pattern}' must not be ignored in .gitignore"


def test_env_example_has_no_real_secrets():
    """Verifies that .env.example contains only safe placeholders without private keys."""
    env_example_path = BASE_DIR / ".env.example"
    assert env_example_path.exists(), ".env.example must exist"

    content = env_example_path.read_text(encoding="utf-8")

    # Check for placeholder markers
    assert "<" in content and ">" in content, ".env.example must contain template placeholder markers"
    assert "ENVIRONMENT=development" in content, ".env.example must specify default environment mode"

    # Ensure no raw 64-character private key is present
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("OPERATOR_PRIVATE_KEY=") or line.startswith("DRONE_PRIVATE_KEY="):
            value = line.split("=", 1)[1].strip()
            assert value.startswith("<") or value.startswith("0x_"), f"Private key field must be a placeholder: {line}"


def test_dataset_yaml_portability():
    """Verifies data/dataset.yaml uses a portable relative path with no hardcoded machine paths."""
    yaml_path = BASE_DIR / "data" / "dataset.yaml"
    assert yaml_path.exists(), "data/dataset.yaml must exist"

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    path_value = str(data.get("path", ""))

    # Must not contain Windows user profiles or absolute OS paths
    assert "C:" not in path_value, "dataset.yaml must not contain hardcoded C: drive path"
    assert "Users" not in path_value, "dataset.yaml must not contain user profile path"
    assert "/home/" not in path_value, "dataset.yaml must not contain absolute home directory path"

    # Must contain unified class taxonomy
    assert "names" in data, "dataset.yaml must define class names"
    assert data["names"][0] == "vehicle"
    assert data["names"][1] == "pedestrian"
    assert data["names"][2] == "cyclist"
    assert data["names"][3] == "traffic_signal"


def test_production_security_validation():
    """Verifies that production mode rejects default or insecure JWT secret keys."""
    # Development mode accepts default for local ease of use
    dev_settings = Settings(
        ENVIRONMENT="development",
        SECRET_KEY="uav-traffic-super-secret-key-change-in-production"
    )
    dev_settings.validate_security()  # Should not raise

    # Production mode MUST reject default/weak keys
    with pytest.raises(ValueError, match="Insecure or default SECRET_KEY is prohibited"):
        prod_settings = Settings(
            ENVIRONMENT="production",
            SECRET_KEY="uav-traffic-super-secret-key-change-in-production"
        )
        prod_settings.validate_security()

    with pytest.raises(ValueError, match="Insecure or default SECRET_KEY is prohibited"):
        prod_settings_short = Settings(
            ENVIRONMENT="production",
            SECRET_KEY="short-secret"
        )
        prod_settings_short.validate_security()

    # Production mode with strong 32+ character key succeeds
    strong_key = "a_very_secure_production_secret_key_32_chars_long!"
    valid_prod_settings = Settings(
        ENVIRONMENT="production",
        SECRET_KEY=strong_key
    )
    valid_prod_settings.validate_security()
    assert valid_prod_settings.SECRET_KEY == strong_key

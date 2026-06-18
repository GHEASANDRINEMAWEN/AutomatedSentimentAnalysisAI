"""Configuration loaded from environment variables.

API credentials are NEVER hard-coded. Set them in your shell or a local .env
file (which is git-ignored). Example (PowerShell):

    setx REDDIT_CLIENT_ID     "your-client-id"
    setx REDDIT_CLIENT_SECRET "your-client-secret"
    setx REDDIT_USER_AGENT    "tourism-sentiment by u/your-username"
"""

import os

# Optionally load a local .env file if python-dotenv is installed.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# --- Reddit (PRAW) ---------------------------------------------------------
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT")


def require(name: str) -> str:
    """Return a required env var, or raise a clear error if it is missing."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in your shell or a .env file (never commit secrets)."
        )
    return value


def validate() -> None:
    """Check that all required credentials are present. Call this at startup."""
    for name in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
        require(name)

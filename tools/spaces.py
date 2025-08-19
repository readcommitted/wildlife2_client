"""
spaces.py — DigitalOcean Spaces Utility Wrapper
-----------------------------------------------

This module provides a simplified interface for interacting with
DigitalOcean Spaces (S3-compatible object storage) using Boto3.

Features:
* Connects to a specific Space using API credentials
* Supports file uploads, downloads, deletions
* Generates signed URLs for secure temporary access
* Provides a utility for downloading to a temp file

Used throughout the Wildlife Vision project to manage image files
stored in the cloud, including raw images, JPEGs, and stage artifacts.

Dependencies:
- boto3
- botocore
- config.settings
- tempfile, pathlib, os

Environment / Settings:
- `SPACE_NAME`: Name of the DigitalOcean space (bucket)
- `REGION`: Region of the space (e.g., "nyc3")
- `ACCESS_KEY`: Spaces API key
- `SECRET_KEY`: Spaces API secret
"""

import boto3
from botocore.client import Config
import os
import tempfile
from pathlib import Path
from config.settings import SPACE_NAME, REGION, ACCESS_KEY, SECRET_KEY

# -------------------------------------------------------------------
# Boto3 Session Setup
# -------------------------------------------------------------------
session = boto3.session.Session()

client = session.client(
    "s3",
    region_name=REGION,
    endpoint_url=f"https://{REGION}.digitaloceanspaces.com",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version="s3v4")
)

# -------------------------------------------------------------------
# Core Utility Functions
# -------------------------------------------------------------------

def list_objects(prefix=""):
    """
    List all object keys in the space under a given prefix.

    Args:
        prefix (str): Folder path or key prefix

    Returns:
        List[str]: Keys (file paths) in the space
    """
    response = client.list_objects_v2(Bucket=SPACE_NAME, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]


def upload_file(local_path, remote_path):
    """
    Uploads a local file to the specified remote path in the space.

    Args:
        local_path (str): Path to the local file
        remote_path (str): Destination key in the space
    """
    client.upload_file(local_path, SPACE_NAME, remote_path)


def delete_file(remote_path):
    """
    Deletes an object from the space.

    Args:
        remote_path (str): Object key to delete
    """
    client.delete_object(Bucket=SPACE_NAME, Key=remote_path)


def generate_signed_url(remote_path, expires_in=3600):
    """
    Generates a signed URL for temporary download access.

    Args:
        remote_path (str): Object key
        expires_in (int): Expiration time in seconds

    Returns:
        str: Pre-signed URL for HTTP GET
    """
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": SPACE_NAME, "Key": remote_path},
        ExpiresIn=expires_in
    )


def download_from_spaces_to_temp(remote_path):
    """
    Downloads a file from Spaces to a temporary location.

    Args:
        remote_path (str): Object key in the space

    Returns:
        Path: Local path to the downloaded temp file
    """
    temp_dir = Path(tempfile.gettempdir())
    local_path = temp_dir / Path(remote_path).name
    client.download_file(SPACE_NAME, remote_path, str(local_path))
    return local_path

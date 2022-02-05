#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 DENOISE - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Managing files access
"""
from pathlib import Path


def get_project_root() -> Path:
    """Get root project folder."""
    return Path(__file__).parent.parent.parent


def get_res_root() -> Path:
    """Get res project folder."""
    return get_project_root() / "res"


def get_keys_root() -> Path:
    """Get key project folder."""
    return get_res_root() / "keys"


def get_data_root() -> Path:
    """Get data project folder."""
    return get_project_root() / "data"


def get_lib_root() -> Path:
    """Get lib project folder."""
    return get_project_root() / "lib"


def get_chromedriver_root() -> Path:
    """Get chromedriver project folder."""
    return get_lib_root() / "chromedriver"


def get_chromedriver_file() -> Path:
    """Get chromedriver file."""
    return get_chromedriver_root() / "chromedriver"


def get_googlechrome_file() -> Path:
    """Get google-chrome path."""
    return Path("/opt/google/chrome/google-chrome")

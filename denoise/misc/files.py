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
    return Path(__file__).parent.parent.parent


def get_res_root() -> Path:
    return get_project_root() / "res"


def get_keys_root() -> Path:
    return get_res_root() / "keys"


def get_data_root() -> Path:
    return get_project_root() / "data"

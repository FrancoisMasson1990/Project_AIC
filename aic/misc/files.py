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


def get_models_root() -> Path:
    """Get models project folder."""
    return get_res_root() / "models"


def get_credentials_root() -> Path:
    """Get credentials project folder."""
    return get_res_root() / "credentials"


def get_data_root() -> Path:
    """Get data project folder."""
    return get_project_root() / "data"


def get_native_root() -> Path:
    """Get native folder."""
    return get_data_root() / "natives"


def get_valve_root() -> Path:
    """Get valve project folder."""
    return get_data_root() / "valve_patient_folder"


def get_dataset_root() -> Path:
    """Get dataset folder."""
    return get_valve_root() / "datasets_dcm"


def get_label_2d_root() -> Path:
    """Get label 2d folder."""
    return get_valve_root() / "labels_2d_npy"


def get_prediction_root() -> Path:
    """Get prediction folder."""
    return get_valve_root() / "predictions"


def get_lib_root() -> Path:
    """Get lib project folder."""
    return get_project_root() / "lib"


def get_configs_root() -> Path:
    """Get config project folder."""
    return get_project_root() / "configs"


def get_valve_credentials() -> Path:
    """Get valve json."""
    return get_credentials_root / "valve-project-332716.json"

#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Library for the files manager
"""

import os
import shutil


def rm_tmp_files(typ=None):
    """Remove tmp file."""
    tmps = ["./cache/prediction.pbz2"]
    for t in tmps:
        if os.path.exists(t):
            if not typ:
                os.remove(t)
            elif t.endswith(typ):
                os.remove(t)


def rm_tmp_folders(typ=None):
    """Remove tmp file."""
    tmps = ["./cache/tmp"]
    for t in tmps:
        if os.path.exists(t):
            shutil.rmtree(t)


def mk_tmp_folder():
    """Generate cache folder."""
    os.makedirs("./cache", exist_ok=True)

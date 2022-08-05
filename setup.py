#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Build-in library
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    # Replace with your own username
    name="aic",
    version="0.0.1",
    author="Francois Masson",
    author_email="francois-masson@hotmail.com",
    description="AIC models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FrancoisMasson1990/Project_AIC",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)

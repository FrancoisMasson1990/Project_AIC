import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aic-models", # Replace with your own username
    version="0.0.1",
    author="Francois Masson",
    author_email="francois-masson@hotmail.com",
    description="AIC models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FrancoisMasson1990/Project_AIC",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
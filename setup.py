import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="denoise",  # Replace with your own username
    version="0.1.0",
    author="Francois Masson",
    author_email="francois-masson@hortmail.com",
    description=("Codes required for the Denoise project"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FrancoisMasson1990/denoise_nft",
    packages=setuptools.find_packages(),
    python_requires='>=3.9',
)

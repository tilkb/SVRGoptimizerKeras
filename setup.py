import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="svrg-optimizer-keras",
    version="0.1.0",
    author="Bence Tilk",
    author_email="bence.tilk@gmail.com",
    description="SVGR optimizer for Keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tilkb/SVRGoptimizerKeras",
    keywords=['SVRG', 'Keras', 'Optimizer'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = [
    'keras>=2.2'
    ]
)


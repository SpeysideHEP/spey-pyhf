from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as f:
    long_description = f.read()

with open("src/spey_pyhf/_version.py", mode="r", encoding="UTF-8") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = ["pyhf==0.7.6", "spey>=0.1.9"]

docs = [
    "sphinx==6.2.1",
    "sphinxcontrib-bibtex~=2.1",
    "sphinx-click",
    "sphinx_rtd_theme",
    "nbsphinx!=0.8.8",
    "sphinx-issues",
    "sphinx-copybutton>=0.3.2",
    "sphinx-togglebutton>=0.3.0",
    "myst-parser",
    "sphinx-rtd-size",
]

setup(
    name="spey-pyhf",
    version=version,
    description=("pyhf plugin for spey interface"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpeysideHEP/spey-pyhf",
    project_urls={
        "Bug Tracker": "https://github.com/SpeysideHEP/spey-pyhf/issues",
        "Documentation": "https://spey-pyhf.readthedocs.io",
        "Repository": "https://github.com/SpeysideHEP/spey-pyhf",
        "Homepage": "https://github.com/SpeysideHEP/spey-pyhf",
        "Download": f"https://github.com/SpeysideHEP/spey-pyhf/archive/refs/tags/v{version}.tar.gz",
    },
    download_url=f"https://github.com/SpeysideHEP/spey-pyhf/archive/refs/tags/v{version}.tar.gz",
    author="Jack Y. Araz",
    author_email=("jackaraz@jlab.org"),
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={
        "spey.backend.plugins": [
            "pyhf.uncorrelated_background = spey_pyhf.interface:UncorrelatedBackground",
            "pyhf = spey_pyhf.interface:FullStatisticalModel",
            "pyhf.simplify = spey_pyhf.simplify:Simplify",
        ]
    },
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    extras_require={
        "dev": ["pytest>=7.1.2", "pytest-cov>=3.0.0", "twine>=3.7.1", "wheel>=0.37.1"],
        "jax": ["jax>=0.3.35"],
        "doc": docs,
    },
)

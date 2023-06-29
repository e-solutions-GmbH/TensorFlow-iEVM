import codecs
import os.path

from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else '"'
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="tf_ievm",
    version=get_version("tf_ievm/__init__.py"),
    description="scikit-learn compatible Extreme Value Machine implementation using TensorFlow operations.",
    url="https://github.com/e-solutions-GmbH/TensorFlow-iEVM",
    author="e.solutions GmbH",
    author_email="info@esolutions.de",
    maintainer="Tobias Koch",
    maintainer_email="27998828+tobokoch@users.noreply.github.com",
    license="MIT License",
    packages=["tf_ievm"],
    python_requires=">=3.6",
    install_requires=["numpy>=1.19.5",
                      "scipy>=1.6",
                      "scikit-learn>=1.2",
                      "tensorflow>=2.4.0"
                      ],
)

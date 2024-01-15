# stdlib
import os

# third party
from setuptools import setup

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def read(fname: str) -> str:
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_version() -> str:
    return "0.0.1"


if __name__ == "__main__":
    try:
        setup(
            version=get_version(),
            author="Severin Elvatun",
            author_email="langberg91@gmail.com",
            description="""Federated flexible parametric models.""",
            long_description=read("README.md"),
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
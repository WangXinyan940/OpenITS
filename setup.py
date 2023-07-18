from setuptools import setup, find_packages

install_requires = []

setup(
    name="openits",
    version="0.1.0",
    author="Wang Xinyan",
    author_email="wangxy940930@gmail.com",
    description=("Implementation of integrated tempering sampling method in OpenMM."),
    url="",
    license="MIT",
    keywords="enhanced sampling",
    install_requires=install_requires,
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
)

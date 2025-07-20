from setuptools import setup, find_packages

setup(
    name='linear_algebra',
    version='0.1.0',
    description='A simple linear algebra library using NumPy',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    python_requires='>=3.7',
)

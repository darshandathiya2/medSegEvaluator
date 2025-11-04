from setuptools import setup, find_packages

setup(
    name='medseg_evaluator',
    version='0.1.0',
    author='Darshan Dathiya',
    author_email='your_email@example.com',
    description='Evaluation toolkit for medical image segmentation models',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'scikit-image',
        'nibabel'
    ],
)

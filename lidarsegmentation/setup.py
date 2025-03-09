from setuptools import setup, find_packages
import os

# Read requirements from requirements_cuda.txt
with open('requirements_cuda.txt') as f:
    requirements = [line.strip() for line in f if not line.startswith('pypcd')]

# Add pypcd separately since it's a git dependency
requirements.append('pypcd @ git+https://github.com/DanielPollithy/pypcd.git@88ab8c98ab81dd620bf4b22d965b37457aab78f8')

# Get long description from README
with open('readme.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="lidarsegmentation",
    version="0.1.0",
    author="LiDAR Segmentation Team",
    description="A package for LiDAR point cloud segmentation and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lidarsegmentation",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'lidarsegmentation': ['settings/*.yaml', 'logo/*'],
    },
    install_requires=requirements,
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'lidarsegmentation=lidarsegmentation.main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 
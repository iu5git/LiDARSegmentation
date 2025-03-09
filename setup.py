from setuptools import setup, find_packages
import sys


def get_requirements():
    # Read requirements from requirements_cuda.txt
    with open('requirements.txt') as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, and platform-specific packages
            if not line or line.startswith('#') or line.startswith('pywin32'):
                continue
    # Add platform-specific dependencies
    if sys.platform.startswith('win'):
        requirements.append('pywin32==306')
        requirements.append('pywin32-ctypes==0.2.2')

    return requirements

def get_long_description():
# Get long description from README
    with open('readme.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
    return long_description

setup(
    name="LiDARSegmentation",
    version="0.1.0",
    description="Locating Trees and Individual Tree Segmentation Using Deep Learning Models LiDAR Based ",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/iu5git/LiDARSegmentation/",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'lidarsegmentation': ['settings/*.yaml', 'logo/*'],
    },
    install_requires=get_requirements(),
    python_requires=">=3.7",
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
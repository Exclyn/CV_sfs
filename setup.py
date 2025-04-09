from setuptools import setup, find_packages

setup(
    name='3d_reconstruction',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'matplotlib',
        'scipy'
    ],
    author='Denzel Lupheng',
    description='A project for 3D reconstruction from single images using shape-from-shading.',
    url='https://github.com/yourusername/3d_reconstruction',
)

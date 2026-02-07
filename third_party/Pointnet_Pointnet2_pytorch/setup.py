from setuptools import setup, find_packages

setup(
    name="pointnet2_pytorch",
    version="0.1.0",        
    author="zhangyiliuxuande",
    description="PointNet/PointNet++ implementation in PyTorch",

    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "tqdm"
    ]
)

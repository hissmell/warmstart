from setuptools import setup, find_packages

setup(
    name="warmstart",
    version="0.1.0",
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        "gdown==4.7.1",
        "torch==1.13.1",
        "matplotlib==3.5.3",
        "scikit-learn==1.0.2",
        "seaborn==0.12.2",
        "pymatgen==2022.0.17",
        "ase==3.22.1",
        "e3fp==1.2.5",
        "bokeh==2.4.3"
        ],
    author="Park won-gyu",
    author_email="snupark@snu.ac.kr",
    description="For warm-start in vasp calculation",
    url="https://github.com/yourusername/ccelpark",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    package_dir={'': '.'}
)
from setuptools import setup, find_packages
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="warmstart",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_dir={
        'warmstart': 'warmstart',
        'warmstart.src': 'warmstart/src'
    },
    install_requirements=required,
    author="Park won-gyu",
    author_email="snupark@snu.ac.kr",
    description="For warm-start in vasp calculation",
    url="https://github.com/hissmell/warmstart",
)
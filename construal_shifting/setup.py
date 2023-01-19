from setuptools import setup, find_packages
setup(
    name='construal_shifting',
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.1',
    description='Construal shifting project',
    author='Mark Ho',
    author_email='mark.ho.cs@gmail.com',
    packages=find_packages(where='.'),
    python_requires='>=3.6, <4'
)

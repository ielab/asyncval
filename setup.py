from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='asyncval',
    version='0.2.0',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    url='https://github.com/ielab/asyncval',
    license='Apache 2.0',
    author='Shengyao Zhuang',
    author_email='s.zhuang@uq.edu.au',
    description='Asyncval: A toolkit for asynchronously validating dense retriever checkpoints during training.',
    python_requires='>=3.7',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "transformers>=4.10.0",
        "datasets>=1.1.3"
    ]
)

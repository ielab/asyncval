from setuptools import setup, find_packages

setup(
    name='asyncval',
    version='0.0.1',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    url='https://github.com/ArvinZhuang/asyncval',
    license='Apache 2.0',
    author='Shengyao Zhuang',
    author_email='s.zhuang@uq.edu.au',
    description='Asyncval: A toolkit for asynchronously validating dense retriever checkpoints during training.',
    python_requires='>=3.7',
    install_requires=[
        "transformers>=4.3.0,<=4.9.2",
        "datasets>=1.1.3",
        "torch>=1.8.0",
        "wandb>=0.12.6",
        "tevatron==0.0.1b1",
        "faiss-cpu==1.7.1",
        "ir-measures>=0.2.3"
    ]
)

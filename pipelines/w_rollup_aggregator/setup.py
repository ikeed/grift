from setuptools import setup, find_packages

setup(
    name='w-rollup-aggregator',
    version='0.1.0',
    description='Dataflow pipeline for windowing and aggregating w-vectors',
    author='GRIFT Team',
    author_email='your-email@example.com',
    packages=find_packages(),
    install_requires=[
        'apache-beam[gcp]>=2.46.0',
        'numpy>=1.24.3',
    ],
    python_requires='>=3.9',
)

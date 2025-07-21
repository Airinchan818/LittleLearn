from setuptools import setup ,find_packages

setup(
    name='littlelearn',
    version='0.1',
    description='machine learning ecosystem',
    long_description=open("README.md", encoding="utf-8").read(),
    author='Candra Alpin Gunawan',
    author_email='hinamatsuriairin@gmail.com',
    url='https://github.com/Airinchan818/LittleLearn',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'numba',
        'matplotlib',
        'networkx'
    ],
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='deep learning,artificial intellengence,engine grad,ai,framework,api,machine learning',
    python_requires='>=3.7',

)
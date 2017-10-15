"""setup file for python module JointInv
"""
import re
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


# get version info from __init__.py
with open("JointInv/__init__.py", 'r') as f:
    version = re.search("(__version__ = )\"(\d\.\d\.\d)\"", f.read()).group(2)

setup(
    name='JointInv',

    # the version across setup.py and the project code, see
    version=version,

    description='A Joint Inversion Softwave for shear wave' 
                'velocity inversion in seismology',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/SeisPider/JointInv',

    # Author details
    author='Xiao Xiao',
    author_email='xiaox.seis@gmail.com',

    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 2 - Pre-Alpha',

        #  
        'Intended Audience :: Developers',
        'Topic :: Seismic Tomography :: Softwave',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3 :: only',
        'Operating System :: POSIX :: Linux',
    ],

    # What does your project relate to?
    keywords='Seismic Tomography',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=['numpy'],

    # $ pip install -e .[dev,test]
    extras_require={
        'dev': [
            'numpy',
            'sklearn',
            'matplotlib.pyplot',
            'obspy',
            'termcolore',
            'PDFPy2',
        ],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'sample': ['package_data.dat'],
    },

    data_files=[('my_data', ['data/*'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'sample=sample:main',
        ],
    },
)

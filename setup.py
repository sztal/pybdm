#!/usr/bin/env python

import os
import sys
from pkg_resources import parse_requirements

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()
doclink = """
Documentation
-------------

The full documentation is at http://pybdm-docs.rtfd.org."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

with open('requirements-dev.txt', 'r') as req:
    tests_require = [ str(r) for r in parse_requirements(req) ]
with open('requirements-build.txt', 'r') as req:
    install_requires = [ str(r) for r in parse_requirements(req) ]

setup(
    name='pybdm',
    version='0.1.0',
    description='Python implementation of block decomposition method for approximating algorithmic complexity.',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Szymon Talaga',
    maintainer='Szymon Talaga',
    maintainer_email='stalaga@protonmail.com',
    url='https://github.com/sztal/pybdm',
    packages=[ *find_packages(exclude=('tests',)) ],
    package_data={'pybdm': ['resources/*.pkl']},
    include_package_data=True,
    setup_requires=['pytest-runner'],
    tests_require=tests_require,
    test_suite='tests',
    install_requires=install_requires,
    license='MIT',
    zip_safe=False,
    keywords=[
        'bdm',
        'ctm',
        'aid',
        'algorithmic information',
        'algorithmic information dynamics',
        'algorithmic complexity',
        'kolmogorov complexity',
        'k-complexity',
        'description length',
        'block decomposition method',
        'coding theorem method',
        'algorithmic information theory'
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)

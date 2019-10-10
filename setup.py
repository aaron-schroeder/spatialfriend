from setuptools import setup, find_packages
import sys


with open('README.md', 'r') as readme_file:
  readme = readme_file.read()

requirements = ['geopy>=1.20.0', 'googlemaps>=3.0', 'numpy>=1.14',
                'pandas>=0.24', 'scipy>=1.1']

setup(name='spatialfriend',
      version='0.0.10',
      author='Aaron Schroeder',
      author_email='aaron@trailzealot.com',
      description='Python library for calculating geospatial data'  \
                + ' from gps coordinates.',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/aaron-schroeder/spatialfriend',
      packages=['spatialfriend'],
      install_requires=requirements,
      extras_require={
        'img': ['GDAL>=2.1.4', 'utm>=0.4.2'],
        'tnm': ['requests>=2.22', 'urllib3>=1.25.6'],
      },
      license='MIT License',
      classifiers=['Programming Language :: Python :: 3.6',
                   'License :: OSI Approved :: MIT License',],)

from setuptools import find_namespace_packages, setup
from pathlib import Path

from configparser import ConfigParser
config = ConfigParser(delimiters=['='])
config.read('settings.ini')
cfg = config['DEFAULT']

dev_requirements = (cfg.get('dev_requirements') or '').split()
requirements = (cfg.get('requirements') or '').split()
long_description = open('README.md').read()


setup(
    name="metaflow-card-notebook",
    version=cfg['version'],
    description="Render Jupyter Notebooks in Metaflow Cards",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hamel Husain",
    author_email="hamel@outerbounds.com",
    license="Apache Software License 2.0",
    packages=find_namespace_packages(include=['metaflow_extensions.*']),
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    extras_require={ 'dev': dev_requirements },
)

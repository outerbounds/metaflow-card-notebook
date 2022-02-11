from setuptools import find_namespace_packages, setup

def get_long_description() -> str:
    with open("README.md") as fh:
        return fh.read()

setup(
    name="metaflow-card-notebook",
    version="1.0.1",
    description="Render Jupyter Notebooks in Metaflow Cards",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Hamel Husain",
    author_email="hamel@outerbounds.com",
    license="Apache Software License 2.0",
    packages=find_namespace_packages(include=['metaflow_extensions.*']),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "ipykernel==6.4.1",
        "papermill==2.3.3",
        "nbconvert==6.4.1",
        "nbformat==5.1.3",
        "metaflow",
    ],
)

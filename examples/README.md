# Examples

This directory contains examples of using the `notebook` card.  Both of these examples involve training a model and visualizing various performance metrics and diagnostics in a Jupyter Notebook as part of your Flow.  The notebook is dynamically updated with the results of the Flow.  There are two different example flows, one that trains a model with Tensorflow and another with a Random Forest.  

Note that we are using [Conda](https://docs.conda.io/en/latest/) for dependency management in these examples.  We understand that not everyone uses Conda, so we have also included a `requirments.txt` file in each directory.  However, we recommend using Conda due to the complex dependencies machine learning libraries often have.

Instructions on running these examples are as follows:

## Deep Learning

1. Setup the environment

    ```bash
    cd deep_learning
    conda env create -f environment.yml
    conda activate mf-demo-dl
    ```

2. Run the flow
    ```bash
    python dl_flow.py --package-suffixes=".ipynb"  run 
    ```

3. View the card

    ```python
    python flow.py card view nb_auto
    ```

---

## Random Forest

1. Setup the environment

    ```bash
    cd random_forest
    conda env create -f environment.yml
    conda activate mf-demo-rf
    ```

 2. Run the flow
    ```bash
    python flow.py --package-suffixes=".ipynb"  run 
    ```

3. View the card

    ```python
    python flow.py card view evaluate
    ```   
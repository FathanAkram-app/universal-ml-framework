Installation
============

.. note::
   Universal ML Framework requires Python 3.7 or higher.

Install from PyPI
-----------------

The easiest way to install Universal ML Framework is using pip:

.. code-block:: bash

   pip install universal-ml-framework

Install from Source
-------------------

For the latest development version:

.. code-block:: bash

   git clone https://github.com/yourusername/universal-ml-framework.git
   cd universal-ml-framework
   pip install -e .

Requirements
------------

Universal ML Framework depends on the following packages:

* **pandas** >= 1.3.0 - Data manipulation and analysis
* **scikit-learn** >= 1.0.0 - Machine learning algorithms
* **numpy** >= 1.21.0 - Numerical computing
* **joblib** >= 1.0.0 - Model persistence

These will be installed automatically when you install the package.

Verify Installation
-------------------

To verify that the installation was successful:

.. code-block:: python

   from universal_ml_framework import UniversalMLPipeline
   print("✅ Installation successful!")

.. tip::
   If you encounter any installation issues, try upgrading pip first:
   
   .. code-block:: bash
   
      pip install --upgrade pip
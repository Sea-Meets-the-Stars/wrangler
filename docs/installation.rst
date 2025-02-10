Installation
============

The wrangler package is currently available directly from GitHub. Here's how to install it:

From Source
----------

To install wrangler from source, clone the repository and install using pip:

.. code-block:: bash

    git clone https://github.com/AI-for-Ocean-Science/wrangler.git
    cd wrangler
    pip install -e .

Dependencies
-----------

Core Dependencies
^^^^^^^^^^^^^^^

* numpy
* pandas
* scipy
* scikit-image
* h5py
* tqdm
* boto3
* smart_open

Data Analysis Dependencies
^^^^^^^^^^^^^^^^^^^^^^^

* seaborn
* matplotlib

Optional Dependencies
^^^^^^^^^^^^^^^^^^

For development and testing:

* pytest
* pytest-cov
* flake8
* black

Environment Setup
---------------

We recommend using a conda environment:

.. code-block:: bash

    # Create a new conda environment
    conda create -n wrangler python=3.9
    
    # Activate the environment
    conda activate wrangler
    
    # Install dependencies
    conda install numpy pandas scipy scikit-image h5py tqdm seaborn matplotlib
    pip install boto3 smart_open
    
    # Install wrangler in development mode
    pip install -e .

AWS/S3 Configuration
------------------

To use wrangler with AWS S3 or compatible storage:

1. Set up your AWS credentials:

   .. code-block:: bash

       # ~/.aws/credentials
       [default]
       aws_access_key_id = YOUR_ACCESS_KEY
       aws_secret_access_key = YOUR_SECRET_KEY

2. Configure the endpoint URL (if using a custom S3 endpoint):

   .. code-block:: bash

       export ENDPOINT_URL='your_endpoint_url'

Troubleshooting
-------------

Common Issues
^^^^^^^^^^^

1. S3 Access Issues:
   
   * Verify your AWS credentials are correctly set up
   * Check your endpoint URL configuration
   * Ensure you have the necessary permissions

2. Missing Dependencies:
   
   * If you encounter import errors, try installing the missing package:
     ``pip install <package_name>``
   * Some features may require optional dependencies

3. Installation Errors:
   
   * Make sure your Python version is 3.7 or higher
   * Try updating pip: ``pip install --upgrade pip``
   * Check that you have the required system libraries installed

Getting Help
----------

If you encounter any issues:

* Check the `GitHub Issues <https://github.com/AI-for-Ocean-Science/wrangler/issues>`_
* Submit a new issue with details about your problem
* Include your Python version and installation environment details
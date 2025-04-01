Field Preprocessing
=================

The ``wrangler.preproc.field`` module provides functionality for preprocessing satellite image fields through a series of customizable operations. This document describes the available preprocessing steps, their parameters, and usage.

Main Preprocessing Function
---------------------------

.. function:: main(field, mask, **kwargs)

   Preprocesses an input field image with a series of optional steps.

   :param field: Input field array 
   :type field: numpy.ndarray
   :param mask: Data mask (True = masked values)
   :type mask: numpy.ndarray
   :param kwargs: Optional preprocessing parameters
   :return: Tuple of (preprocessed_field, metadata_dict)
   :rtype: tuple

Preprocessing Steps
-------------------

The preprocessing steps are applied in the following order if enabled through parameters:

1. **Inpainting**
   
   Fill masked regions using a biharmonic inpainting algorithm.

   :param inpaint: Enable inpainting
   :type inpaint: bool
   :param only_inpaint: Return after inpainting
   :type only_inpaint: bool

2. **Resizing**
   
   Resize the field to a specified size.

   :param resize: Enable resizing 
   :type resize: bool
   :param field_size: Target size for the field
   :type field_size: int

3. **Noise Addition**
   
   Add Gaussian noise to the field.

   :param noise: Standard deviation of noise to add
   :type noise: float

4. **Median Filtering**
   
   Apply a median filter to reduce noise while preserving edges.

   :param median: Enable median filtering
   :type median: bool
   :param med_size: Size of the median filter window (tuple)
   :type med_size: tuple

5. **Downscaling**
   
   Reduce the field size by local mean downscaling.

   :param downscale: Enable downscaling
   :type downscale: bool
   :param dscale_size: Downscaling factors (tuple)
   :type dscale_size: tuple

6. **Mean Removal**
   
   Normalize the field by removing the mean value.

   :param de_mean: Enable mean removal
   :type de_mean: bool
   :param min_mean: Minimum required mean value
   :type min_mean: float

7. **Sigmoid Transformation**
   
   Apply error function (erf) transformation to field values.

   :param sigmoid: Enable sigmoid transformation
   :type sigmoid: bool

8. **Scaling**
   
   Apply a multiplicative scaling factor to field values.

   :param scale: Scaling factor
   :type scale: float

9. **Exponentiation**
   
   Raise field values to a power, preserving sign.

   :param expon: Exponent value
   :type expon: float

10. **Gradient Enhancement**
    
    Apply a Sobel filter to enhance edges and gradients.

    :param gradient: Enable gradient enhancement
    :type gradient: bool

11. **Logarithmic Scaling**
    
    Apply logarithmic scaling to gradient values.

    :param log_scale: Enable logarithmic scaling
    :type log_scale: bool

Metadata
--------

The function returns a metadata dictionary containing statistics about the field:

- ``Tmax``: Maximum temperature value
- ``Tmin``: Minimum temperature value
- ``T10``: 10th percentile temperature
- ``T90``: 90th percentile temperature
- ``mu``: Mean field value

If gradient enhancement is enabled, additional metadata is included:

- ``G10``: 10th percentile gradient value
- ``G90``: 90th percentile gradient value
- ``Gmax``: Maximum gradient value

Usage Examples
--------------

Basic Inpainting
^^^^^^^^^^^^^^^^

.. code-block:: python

    from wrangler.preproc.field import main as process_field
    
    # Load field and mask data
    # ...
    
    # Just inpaint the field
    inpainted_field, _ = process_field(field, mask, inpaint=True, only_inpaint=True)

Multiple Preprocessing Steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Apply several preprocessing steps
    processed_field, metadata = process_field(
        field, 
        mask,
        inpaint=True,
        median=True,
        med_size=(3, 1),
        downscale=True,
        dscale_size=(2, 2),
        de_mean=True,
        scale=0.1
    )
    
    print(f"Mean value: {metadata['mu']}")
    print(f"Temperature range: {metadata['Tmin']} to {metadata['Tmax']}")

Gradient Enhancement Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create a gradient-enhanced version
    gradient_field, metadata = process_field(
        field,
        mask,
        inpaint=True,
        de_mean=True,
        gradient=True,
        log_scale=True
    )
    
    print(f"Gradient range: {metadata['G10']} to {metadata['G90']}")

Important Notes
---------------

- Not all combinations of parameters are valid. For example, ``log_scale=True`` requires ``gradient=True``.
- The function will return ``None, None`` if any of these conditions occur:
  - Inpainting fails
  - The field contains NaN values after processing
  - The mean is below the ``min_mean`` threshold
- For resizing, the field is resized to match the specified ``field_size``
- When using ``expon``, the sign of values is preserved (negative values remain negative)

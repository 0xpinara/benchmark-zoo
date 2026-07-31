"""bzoo: multiplicity corrections with an empirically calibrated null.

The package has two halves that share one statistical core:

``bzoo.finance``  the validation testbed, where a large population of
                  strategies is null by construction, so a calibration can
                  be checked against ground truth;
``bzoo.ml``       the application testbed, where the null has to be
                  constructed but the number of trials is partly observable.

``bzoo.corrections`` holds the corrections, ``bzoo.null`` the calibration,
and ``bzoo.resample`` the two resampling schemes the two domains need.
"""

__version__ = "0.1.0"

MiraPy: Python Package for Deep Learning in Astronomy
--------------------------------------------------------

.. image:: https://img.shields.io/badge/Powered%20by-Keras-red?style=flat-square
    :target: http://keras.io
    :alt: Powered by Keras Badge

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat-square
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

.. image:: https://img.shields.io/travis/com/mirapy-org/mirapy.svg?style=flat-square&logo=travis%20ci
    :target: https://travis-ci.com/mirapy-org/mirapy
    :alt: Travis CI

.. image:: https://readthedocs.org/projects/mirapy/badge/?version=latest&style=flat-square
    :target: https://mirapy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/coveralls/github/mirapy-org/mirapy.svg?style=flat-square
    :target: https://coveralls.io/github/mirapy-org/mirapy
    :alt: Coveralls

.. image:: https://img.shields.io/badge/chat%20on-Slack-4A154B.svg?style=flat-square&logo=slack
    :target: https://join.slack.com/t/mirapy/shared_invite/enQtNjEyNDQwNTI2NDY3LTE3ZmI3M2EyMjdkZWU4NTE2NjkxZjdhYWE4ZjUyODY0NzllNzRlMzZhNThhNWRiMjk4MjNhYWQ3NjA3YjJiNGY
    :alt: Slack

.. image:: https://img.shields.io/pypi/v/mirapy.svg?style=flat-square&logo=pypi
    :target: https://pypi.org/project/mirapy/
    :alt: PyPI

.. image:: https://img.shields.io/github/license/mirapy-org/mirapy.svg?style=flat-square
    :target: https://github.com/mirapy-org/mirapy/blob/master/LICENSE.rst
    :alt: LICENSE

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2908315.svg
    :target: https://doi.org/10.5281/zenodo.2908315
    :alt: Zenodo DOI


MiraPy is a Python package for Deep Learning in Astronomy. It is built using
Keras for developing ML models to run on CPU and GPU seamlessly. The
aim is to make applying machine learning techniques on astronomical data easy
for astronomers, researchers and students.

The documentation is available `here <https://mirapy.readthedocs.io>`_.

Applications
------------

MiraPy can be used for problem solving using ML techniques and will continue to grow to tackle new problems in Astronomy. Following are some of the experiments that you can perform right now:

- Classification of X-Ray Binaries using neural network
- Astronomical Image Reconstruction using Autoencoder
- Classification of the first catalog of variable stars by ATLAS
- HTRU1 Pulsar Dataset Image Classification using Convolutional Neural Network
- OGLE Catalogue Variable Star Classification using Recurrent Neural Network (RNN)
- 2D and 3D visualization of feature sets using Principal Component Analysis (PCA)
- Curve Fitting using Autograd (basic implementation)

There are more projects that we will add soon and some of them are as following:

- Feature Engineering (Selection, Reduction and Visualization)
- Classification of different states of GRS1905+105 X-Ray Binaries using Recurrent Neural Network (RNN)
- Feature extraction from Images using Autoencoders and its applications in Astronomy

You can find the applications MiraPy in our `tutorial <https://github.com/mirapy-org/tutorials>`_ repository.

Installation
------------

Before installing Keras, please install one of its backend engines: TensorFlow, Theano, or CNTK. We recommend the TensorFlow backend. You can find Keras installation guide `here <https://keras.io/#installation>`_.

You can download the package using `pip` package installer::

    pip install mirapy

You can also build from source code::

    git clone --recursive https://github.com/mirapy-org/mirapy.git
    cd mirapy
    pip install -r requirements.txt
    python setup.py install

Contributing
------------

MiraPy is far from perfect and we would love to see your contributions to open source community! In future, it will be able to do more and in better ways and we need your suggestions! Tell us what you would like to see as a part of this package on `Slack <https://join.slack.com/t/mirapy/shared_invite/enQtNjEyNDQwNTI2NDY3LTE3ZmI3M2EyMjdkZWU4NTE2NjkxZjdhYWE4ZjUyODY0NzllNzRlMzZhNThhNWRiMjk4MjNhYWQ3NjA3YjJiNGY>`_.


About Us
--------

MiraPy is developed by `Swapnil Sharma <https://www.linkedin.com/in/swapsha96/>`_ and `Akhil Singhal <https://www.linkedin.com/in/akhil-singhal-a59448106/>`_ as their final year 'Major Technical Project' under the guidance of `Dr. Arnav Bhavsar <http://faculty.iitmandi.ac.in/~arnav/>`_ at `Indian Institute of Technology, Mandi <http://iitmandi.ac.in/>`_.

License
-------

This project is Copyright (c) Swapnil Sharma, Akhil Singhal and licensed under
the terms of the MIT license.

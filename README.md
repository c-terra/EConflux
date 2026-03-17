# EConflux

A library for binning, geolocation, processing, and statistical analysis of multi-method geophysical data.

EConflux is designed to support users conducting joint geophysical investigations by providing functions to assist with georeferencing, co-locating, processing, and interpreting complementary geophysical datasets.

The current version of EConflux is designed to be incorporated into workflows using ResIPy (https://gitlab.com/hkex/resipy) and EMagPy (https://gitlab.com/hkex/emagpy) but should provide a suitable foundation for users looking to incorporate the functions into other inversion codes.

## Installing EConflux

EConflux is available for installation on PyPI using pip or can be installeld locally by cloning the repository and perfomring a local pip install. Both options are outlined here as well as how to install just the required dependencies or the required dependencies plus optional packages needed to run the Jupyter Notebook tutorials.

### Installing EConflux Using PyPI

EConflux can be installed with just its required dependencies by using:

```
pip install econflux
```
EConflux with its required dependencies plus the additional packages that are not needed for EConflux to run but are necessary to work through the example Jupyter Notebooks can be installed by using:

```
pip install econflux[examples]
```

### Installing EConflux locally

Navigate to a preferred directory:
```
cd \path\to\clone\location
```
Clone the EConflux git repository
```
git clone https://github.com/c-terra/EConflux
```
EConflux can then be used by adding the path to the cloned EConflux repository to your Python sys path (see how this is done in the Jupyter Notebooks in the "examples" directory), or it can be installed using PyPI locally.
This is done by doing:

```
cd \path\to\EConflux
pip install .
```
Otherwise, EConflux, its dependencies, and the needed packages to run the Jupyter Notebook examples can be installed locally using:

```
cd \path\to\EConflux
pip install .[examples]
```

## Overview of EConflux Classes and Structure

<img width="7089" height="4438" alt="econflux_flowchart" src="https://github.com/user-attachments/assets/54ce2f9d-de86-4e2d-97d8-501528849fe4" />

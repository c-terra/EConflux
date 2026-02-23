import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name = "econflux",
    version = "0.1.0",
    author = "Chris Terra",
    author_email = "cterra413@gmail.com",
    description = "Workflows for multi-method geophysical investigations",
    url="https://github.com/c-terra/EConflux",
    license="MIT License",
    classifiers = [
        "Development Status :: 4 - Beta",
    	"Intended Audience :: Developers",
    	"Intended Audience :: Science/Research",
  	    "License :: OSI Approved :: MIT License",
   	    "Programming Language :: Python",
  	    "Topic :: Scientific/Engineering",
 	    "Topic :: Scientific/Engineering :: Mathematics",
  	    "Operating System :: OS Independent",
    	"Natural Language :: English",
]
    python_requires = '>=3.11',
    install_requires=required,
)

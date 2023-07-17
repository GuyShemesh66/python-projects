from setuptools import setup, Extension

# Define the extension module
kmeans_extension = Extension('mykmeanssp',sources=['kmeansmodule.c'])

# Setup configuration
setup(name='mykmeanssp',version='1.0',description='Extension module for K-means algorithm',ext_modules=[kmeans_extension])

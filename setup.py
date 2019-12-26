from setuptools import setup

setup(
   name='jax_pd',
   version='1.0.0',
   description='Peridynamics implemented in JAX',
   author='Ravinder Bhattoo',
   author_email='',
   packages=['jax_pd'],
   install_requires=['jax', 'jax_md', 'matplotlib']
)

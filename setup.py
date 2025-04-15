from setuptools import setup, find_packages

setup(
    name="ECUALI_NOLI",
    version="0.1",
    packages=find_packages(),
    description="Paquete para resolver ecuaciones lineales y no lineales",
    author="Tu Nombre",
    author_email="tu@email.com",
    install_requires=["numpy", "scipy"],
)
from setuptools import setup
setup(
    name="ECUALI_NOLI",
    version="0.1",
    packages=['ECUALI_NOLI'],
    description="Paquete para resolver ecuaciones lineales y no lineales",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    author="Kenner Melgar",
    author_email="NM23016@ues.edu.sv",
    url= 'https://github.com/Kenner23016/Ecuaciones.git'
    install_requires=["numpy", "scipy", "copy"],
)
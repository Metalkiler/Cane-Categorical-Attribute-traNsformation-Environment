import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='cane',
                 version='0.0.4.1',
                 description='Categorical Arrangement of Nominal variables Environment (CANE)',
                 author='Luís Miguel Matos, Paulo Cortez, Rui Mendes',
                 license='MIT',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 author_email='luis.matos@dsi.uminho.pt',
                 packages=setuptools.find_packages(),
                 install_requires=['numpy', 'pandas'],
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ],
                 python_requires='>=3.6')

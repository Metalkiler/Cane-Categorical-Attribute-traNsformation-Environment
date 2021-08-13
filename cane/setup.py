import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='cane',
                 version='2.0.4.1',
                 description='Cane - Categorical Attribute traNsformation Environment',
                 author='Luís Miguel Matos, Paulo Cortez, Rui Mendes',
                 license='MIT',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 author_email='luis.matos@dsi.uminho.pt',
                 packages=setuptools.find_packages(),
                 install_requires=["bounded-pool-executor","numpy","pandas","pqdm","python-dateutil","pytz","tqdm","typing-extensions","pqdm","sklearn"],
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ],
                 url="https://github.com/Metalkiler/Cane-Categorical-Attribute-traNsformation-Environment",
                 python_requires='>=3.8')

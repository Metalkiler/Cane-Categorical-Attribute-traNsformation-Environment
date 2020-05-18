import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='pyprocessing',
                 version='0.1',
                 description='A simpler preprocessing method for machine learning',
                 author=['Luís Miguel Matos', 'Paulo Cortez', 'Rui Mendes'],
                 url='http://github.com/storborg/funniest',
                 license='MIT',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 author_email='luis.matos@dsi.uminho.pt',
                 packages=setuptools.find_packages(),
                 install_requires=['numpy', "pandas"],
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ],
                 python_requires='>=3.6', )

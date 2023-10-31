from setuptools import setup, find_packages 
from typing import List 

def get_requirements(filename): 
    '''
    args: 
        filename: name of the file containing the list of requirements
    returns:
        list of requirements
    summary: 
        reads the file and returns the list of requirements         
    
    '''
    requirements = [] 
    with open(filename, 'r') as f:
        for line in f.readlines():
            requirements.append(line.strip())
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

#setup 
setup( 
    name='Drondata_Segmentation', 
    version='1.0', 
    packages=find_packages(), 
    include_package_data=True, 
    install_requires=get_requirements('requirements.txt'),
    author='Amzad Rafi',
    author_email='amzad.rafi@northsouth.edu',
    description='Drondataset Segmentation for segmentation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
  
)   
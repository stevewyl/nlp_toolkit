from setuptools import setup, find_packages

with open('requirements.txt') as requirements:
    REQUIREMENTS = requirements.readlines()
long_description = open('README.md', encoding='utf-8').read()

REQUIREMENTS = ['seqeval>=0.0.5', 'Keras>=2.2.4',
                'tensorflow>=1.9.0', 'jieba>=0.39',
                'numpy>=1.14.3', 'scikit-learn>=0.19.1',
                'hanziconv>=0.3.2', 'ruamel.yaml>=0.15.81']

setup(
    name='nlp_toolkit',
    version='1.3.0',
    description='NLP Toolkit with easy model training and applications',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='yilei.wang',
    author_email='stevewyl@163.com',
    license='MIT',
    install_requires=REQUIREMENTS,
    extra_requires={
        'tensorflow_gpu': ['tensorflow-gpu>=1.10.0'],
        'GPUtil': ['GPUtil>=1.3.0'],
    },
    python_requires='>=3.6',
    packages=find_packages(),
    package_data={'nlp_toolkit': ['data/*.txt']},
    include_package_data=True,
    url='https://github.com/stevewyl/nlp_toolkit',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='nlp keras text classification sequence labeling',
)

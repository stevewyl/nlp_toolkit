from setuptools import setup, find_packages

with open('requirements.txt') as requirements:
    REQUIREMENTS = requirements.readlines()
long_description = open('README.md', encoding='utf-8').read()

REQUIREMENTS = ['seqeval>=0.0.5', 'Keras>=2.2.4',
                'tensorflow>=1.9.0', 'jieba>=0.39',
                'numpy>=1.14.3', 'scikit-learn>=0.19.1',
                'hanziconv>=0.3.2']

setup(
    name='nlp_toolkit',
    version='1.0.8',
    description='NLP Toolkit with easy model training and applications',
    long_description=long_description,
    author='yilei.wang',
    author_email='stevewyl@163.com',
    license='MIT',
    install_requires=REQUIREMENTS,
    python_requires='>=3.6',
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/stevewyl/nlp_toolkit',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ]
)

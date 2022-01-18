from setuptools import find_packages, setup


def _safe_read_lines(f):
    with open(f) as in_f:
        r = in_f.readlines()
    r = [l.strip() for l in r]
    return r


install_requires = _safe_read_lines("./requirements.txt")


classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Natural Language :: English',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]


def description():
    description = (
        "This package implements TCR-peptide interaction prediction with AVIB and MVIB."
    )
    return description


setup(
    name='vibtcr',
    version='0.1.0',
    description=description(),
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=True,
    classifiers=classifiers,
)

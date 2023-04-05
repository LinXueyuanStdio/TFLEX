from setuptools import setup


# Requirements used for submodules
cli = [
    'tqdm',
    'numpy',
    'scikit-learn',
    'scipy',
    'click',
    'tensorboardX',
    'pandas',
    'pathlib',
    'pyyaml',
]

develop = [
    'coveralls',
    'flake8',
    'flake8-tidy-imports',
    'mypy',
    'pytest',
    'pytest-asyncio',
    'pytest-cov',
    'pytest-mock',
    'pytest-random-order',
]

all_extra = develop + cli

setup(
    tests_require=[
        'pytest',
        'pytest-asyncio',
        'pytest-cov',
        'pytest-mock',
    ],
    install_requires=[
        # from requirements.txt
        'tqdm',
        'numpy',
        'scikit-learn',
        'scipy',
        'click',
        'tensorboardX',
        'pandas',
        'pathlib',
        'pyyaml',
    ],
    extras_require={
        'dev': all_extra,
        'cli': cli,
        'all': all_extra,
    },
)

from setuptools import find_packages, setup

install_requires = ["numpy",
                    "scipy",
                    "six"]


setup_requires = ['pytest-runner']


tests_require = ["pytest",
                 "pytest-cov",
                 "mock"]


keywords = ["shapley",
            "random-forest",
            "data-science",
            "shap",
            "ensemble",
            "expert-system",
            "explanability",
            "voting-classifier",
            "classifier",
            "machine-learning",
            "deep-learning",
            "deeplearning"]


setup(
  name = "shapley",
  packages = find_packages(),
  version = "0.0.1",
  license = "MIT",
  description = "A general purpose library for quantify the value of classifiers in a machine learning ensemble.",
  author = "Benedek Rozemberczki",
  author_email = "benedek.rozemberczki@gmail.com",
  url = "https://github.com/benedekrozemberczki/shapley",
  download_url = "https://github.com/benedekrozemberczki/shapley/archive/v_00001.tar.gz",
  keywords = keywords,
  install_requires = install_requires,
  setup_requires = setup_requires,
  tests_require = tests_require,
  classifiers = ["Development Status :: 3 - Alpha",
                 "Intended Audience :: Developers",
                 "Topic :: Software Development :: Build Tools",
                 "License :: OSI Approved :: MIT License",
                 "Programming Language :: Python :: 3.7"],
)

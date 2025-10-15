import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    packages = [
        "pandas",
        "matplotlib",
        "seaborn",
        "numpy",
        "pyreadr",
        "nltk",
        "torch",
        "transformers",
        "scikit-learn",
        "tqdm",
        "wordcloud",
        "spacy"
    ]

    for package in packages:
        install_package(package)

if __name__ == "__main__":
    main()
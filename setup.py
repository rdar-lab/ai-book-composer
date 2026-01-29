from setuptools import setup, find_packages

setup(
    name="ai-book-composer",
    version="0.1.0",
    description="Using AI and Deep Agent pattern to generate a book based on list of articles/files",
    author="",
    author_email="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "ai-book-composer=ai_book_composer.cli:main",
        ],
    },
    python_requires=">=3.9",
)

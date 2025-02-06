from setuptools import setup, find_packages

setup(
    name="gym_pathfinding",  # Name of the package
    version="0.1.0",  # Package version
    description="A reinforcement learning environment for pathfinding in 2D space",  # Short description of the package
    long_description=open("README.md").read(),  # Long description, typically extracted from the README file
    long_description_content_type="text/markdown",  # Specify the content type of the long description
    author="Your Name",  # Author of the package
    author_email="youremail@example.com",  # Email address of the author
    url="https://github.com/yourusername/gym_pathfinding",  # URL for the project (GitHub link, etc.)
    packages=find_packages(),  # Automatically find all packages in the project
    install_requires=[  # List of external dependencies that your package needs
        "numpy>=1.21.0",  # Ensure the project works with this version or higher of numpy
        "pygame>=2.0.0",  # For visualization
        "gym>=0.18.0",  # OpenAI Gym
    ],
    extras_require={  # Optional additional dependencies
        "dev": [
            "pytest>=6.0",  # For running tests
            "black",  # For code formatting
        ],
    },
    classifiers=[  # Classifiers for PyPI (Python Package Index)
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # License type
        "Operating System :: OS Independent",  # OS compatibility
    ],
    python_requires=">=3.7",  # Specifies the minimum required Python version
)

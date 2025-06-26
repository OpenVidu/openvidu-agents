from setuptools import setup, find_packages

setup(
    name="openviduagentutils",
    version="0.0.1",
    description="Helper utilities for OpenVidu AI Agents",
    url="https://github.com/openvidu/openvidu-agents",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.9.0",
    install_requires=["python-dotenv>=1.1.0", "pyyaml>=6.0.2"],
    author="Pablo Fuente",
    author_email="pablofuenteperez@gmail.com",
    keywords=["openvidu", "agents", "ai"],
    project_urls={
        "Documentation": "https://openvidu.io/latest/docs/getting-started/",
        "Website": "https://openvidu.io/",
        "Source": "https://github.com/openvidu/openvidu-agents",
    },
)

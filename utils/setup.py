import pathlib

import setuptools
import setuptools.command.build_py

here = pathlib.Path(__file__).parent.resolve()
about = {}

setuptools.setup(
    name="openvidu-agents-utils",
    version="1.0.0",
    description="Helper utilities for OpenVidu agents",
    url="https://github.com/openvidu/openvidu-agents",
    cmdclass={},
    license="Apache-2.0",
    python_requires=">=3.9.0",
    install_requires=["redis>=5.2.1", "python-dotenv>=1.0.1"],
    package_data={"openvidu.agents.utils": ["py.typed"]},
    project_urls={
        "Documentation": "https://openvidu.io/latest/docs/getting-started/",
        "Website": "https://openvidu.io/",
        "Source": "https://github.com/openvidu/openvidu-agents",
    },
)

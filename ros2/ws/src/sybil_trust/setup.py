from setuptools import setup

setup(
    name="sybil_trust",
    version="0.1.0",
    packages=["sybil_trust"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/sybil_trust"]),
        ("share/sybil_trust", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": ["trust_node = sybil_trust.trust_node:main"]},
)

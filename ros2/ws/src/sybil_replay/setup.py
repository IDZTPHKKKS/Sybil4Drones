from glob import glob

from setuptools import setup

setup(
    name="sybil_replay",
    version="0.1.0",
    packages=["sybil_replay"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/sybil_replay"]),
        ("share/sybil_replay", ["package.xml"]),
        ("share/sybil_replay/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "sim_to_bag = sybil_replay.sim_to_bag:main",
        "checker = sybil_replay.checker:main",
        "viz = sybil_replay.viz:main",
        "gate = sybil_replay.gate:main",
        "world_map = sybil_replay.world_map:main",
        "voxel_map = sybil_replay.voxel_map:main",
    ]},
)

from setuptools import find_packages, setup

package_name = "vlm_webcam_demo"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="VLMshowcase",
    description="ROS2 webcam demo for VLMshowcase",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vlm_webcam_node = vlm_webcam_demo.webcam_node:main",
        ],
    },
)

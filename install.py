import launch
import os
import pkg_resources
import subprocess
from typing import Tuple, Optional

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")


def comparable_version(version: str) -> Tuple:
    return tuple(version.split('.'))
    

def get_installed_version(package: str) -> Optional[str]:
    try:
        return pkg_resources.get_distribution(package).version
    except Exception:
        return None


def pip_install(pkg_name, update=True):
    if update:
        subprocess.check_call([launch.python, "-m", "pip", "install", "-U", pkg_name, "--prefer-binary"])
    else:
        subprocess.check_call([launch.python, "-m", "pip", "install", pkg_name, "--prefer-binary"])



# install requirements
with open(req_file) as file:
    for package in file:
        try:
            package = package.strip()
            if '==' in package:
                package_name, package_version = package.split('==')
                installed_version = get_installed_version(package_name)
                if installed_version != package_version:
                    pip_install(package, update=True)
                    print(f"sd-webui-prettyu requirement: changing {package_name} version from {installed_version} to {package_version}")
                    # launch.run_pip(f"install -U {package}", f"sd-webui-prettyu requirement: changing {package_name} version from {installed_version} to {package_version}")
            elif '>=' in package:
                package_name, package_version = package.split('>=')
                installed_version = get_installed_version(package_name)
                if not installed_version or comparable_version(installed_version) < comparable_version(package_version):
                    pip_install(package, update=True)
                    print(f"sd-webui-prettyu requirement: changing {package_name} version from {installed_version} to {package_version}")
                    # launch.run_pip(f"install -U {package}", f"sd-webui-prettyu requirement: changing {package_name} version from {installed_version} to {package_version}")
            elif not launch.is_installed(package):
                pip_install(package, update=False)
                print(f"sd-webui-prettyu requirement: {package}")
                # launch.run_pip(f"install {package}", f"sd-webui-prettyu requirement: {package}")
        except Exception as e:
            print(e)
            print(f'Warning: Failed to install {package}, some preprocessors may not work.')

# install kohya
kohya_dir = os.path.join(cur_dir, 'prettyu', 'third_party', 'sd-scripts')
try:
    subprocess.check_call([launch.python, "-m", "pip", "install", kohya_dir])
except subprocess.CalledProcessError as e:
    print(f"Couldn't install kohya on repository in '{kohya_dir}':\n{e.output.decode('utf-8').strip()}\n")

import sys
import subprocess
import pkg_resources

required = {'pandas', 'numpy', 'matplotlib', 'sklearn', 'seaborn', 'parallel-execute', 'tensorflow'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

else:
	print("all required packages are already installed.")

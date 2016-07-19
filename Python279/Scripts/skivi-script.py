#!C:\Python279\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'scikit-image==0.11.3','console_scripts','skivi'
__requires__ = 'scikit-image==0.11.3'
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.exit(
        load_entry_point('scikit-image==0.11.3', 'console_scripts', 'skivi')()
    )

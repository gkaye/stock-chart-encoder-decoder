#!C:\cb\PYTORC~1\_h_env\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'torch==1.5.0','console_scripts','convert-onnx-to-caffe2'
__requires__ = 'torch==1.5.0'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('torch==1.5.0', 'console_scripts', 'convert-onnx-to-caffe2')()
    )

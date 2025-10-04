# forerroronly
ahmadradhy@ubuntu:~/onnxruntime$ pip install "cmake==3.10.3"
Defaulting to user installation because normal site-packages is not writeable
Collecting cmake==3.10.3
  Downloading cmake-3.10.3.tar.gz (27 kB)
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [6 lines of output]
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 35, in <module>
        File "/tmp/pip-install-94mijgo3/cmake_57f6f7cfa8334ce9a41fe027b765f358/setup.py", line 7, in <module>
          from skbuild import setup
      ModuleNotFoundError: No module named 'skbuild'
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.

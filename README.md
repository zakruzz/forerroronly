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

[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
image = "0.25"
imageproc = "0.25"
ndarray = { version = "0.15", features = ["rayon"] }
noise = "0.9"
num-traits = "0.2"
nokhwa = { version = "0.10", features = ["input-v4l", "output-threaded"] }
ort = { version = "2", features = ["download-binaries", "cuda"], default-features = false }
rayon = "1"
thiserror = "1"

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.

error: failed to run custom build command for `opencv v0.90.0`

Caused by:
  process didn't exit successfully: `/home/ahmadradhy/tugasakhir-fairuz/yolo-trt-rs/target/debug/build/opencv-111c95cb821fa458/build-script-build` (exit status: 101)
  --- stdout
  cargo:rerun-if-env-changed=OPENCV4_NO_PKG_CONFIG
  cargo:rerun-if-env-changed=PKG_CONFIG_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG
  cargo:rerun-if-env-changed=PKG_CONFIG
  cargo:rerun-if-env-changed=OPENCV4_STATIC
  cargo:rerun-if-env-changed=OPENCV4_DYNAMIC
  cargo:rerun-if-env-changed=PKG_CONFIG_ALL_STATIC
  cargo:rerun-if-env-changed=PKG_CONFIG_ALL_DYNAMIC
  cargo:rerun-if-env-changed=PKG_CONFIG_PATH_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_PATH_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG_PATH
  cargo:rerun-if-env-changed=PKG_CONFIG_PATH
  cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG_LIBDIR
  cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR
  cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG_SYSROOT_DIR
  cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR
  cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR
  cargo:rerun-if-env-changed=SYSROOT
  cargo:rerun-if-env-changed=OPENCV4_STATIC
  cargo:rerun-if-env-changed=OPENCV4_DYNAMIC
  cargo:rerun-if-env-changed=PKG_CONFIG_ALL_STATIC
  cargo:rerun-if-env-changed=PKG_CONFIG_ALL_DYNAMIC
  cargo:rerun-if-env-changed=PKG_CONFIG_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG
  cargo:rerun-if-env-changed=PKG_CONFIG
  cargo:rerun-if-env-changed=OPENCV4_STATIC
  cargo:rerun-if-env-changed=OPENCV4_DYNAMIC
  cargo:rerun-if-env-changed=PKG_CONFIG_ALL_STATIC
  cargo:rerun-if-env-changed=PKG_CONFIG_ALL_DYNAMIC
  cargo:rerun-if-env-changed=PKG_CONFIG_PATH_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_PATH_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG_PATH
  cargo:rerun-if-env-changed=PKG_CONFIG_PATH
  cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG_LIBDIR
  cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR
  cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG_SYSROOT_DIR
  cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR
  cargo:rustc-cfg=ocvrs_opencv_branch_4

  --- stderr
  === Crate version: Some("0.90.0")
  === Environment configuration:
  ===   OPENCV_PACKAGE_NAME = None
  ===   OPENCV_PKGCONFIG_NAME = None
  ===   OPENCV_CMAKE_NAME = None
  ===   OPENCV_CMAKE_BIN = None
  ===   OPENCV_VCPKG_NAME = None
  ===   OPENCV_LINK_LIBS = None
  ===   OPENCV_LINK_PATHS = None
  ===   OPENCV_INCLUDE_PATHS = None
  ===   OPENCV_DISABLE_PROBES = None
  ===   OPENCV_MSVC_CRT = None
  ===   CMAKE_PREFIX_PATH = None
  ===   OpenCV_DIR = None
  ===   PKG_CONFIG_PATH = None
  ===   VCPKG_ROOT = None
  ===   VCPKGRS_DYNAMIC = None
  ===   VCPKGRS_TRIPLET = None
  ===   OCVRS_DOCS_GENERATE_DIR = None
  ===   DOCS_RS = None
  ===   PATH = Some("/usr/local/cuda-12.6/bin:/home/ahmadradhy/.cargo/bin:/home/ahmadradhy/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin")
  === Enabled features:
  ===   ALPHAMAT
  ===   ARUCO
  ===   ARUCO_DETECTOR
  ===   BARCODE
  ===   BGSEGM
  ===   BIOINSPIRED
  ===   CALIB3D
  ===   CCALIB
  ===   CUDAARITHM
  ===   CUDABGSEGM
  ===   CUDACODEC
  ===   CUDAFEATURES2D
  ===   CUDAFILTERS
  ===   CUDAIMGPROC
  ===   CUDAOBJDETECT
  ===   CUDAOPTFLOW
  ===   CUDASTEREO
  ===   CUDAWARPING
  ===   CVV
  ===   DEFAULT
  ===   DNN
  ===   DNN_SUPERRES
  ===   DPM
  ===   FACE
  ===   FEATURES2D
  ===   FLANN
  ===   FREETYPE
  ===   FUZZY
  ===   GAPI
  ===   HDF
  ===   HFS
  ===   HIGHGUI
  ===   IMGCODECS
  ===   IMGPROC
  ===   IMG_HASH
  ===   INTENSITY_TRANSFORM
  ===   LINE_DESCRIPTOR
  ===   MCC
  ===   ML
  ===   OBJDETECT
  ===   OPTFLOW
  ===   OVIS
  ===   PHASE_UNWRAPPING
  ===   PHOTO
  ===   PLOT
  ===   QUALITY
  ===   RAPID
  ===   RGBD
  ===   SALIENCY
  ===   SFM
  ===   SHAPE
  ===   STEREO
  ===   STITCHING
  ===   STRUCTURED_LIGHT
  ===   SUPERRES
  ===   SURFACE_MATCHING
  ===   TEXT
  ===   TRACKING
  ===   VIDEO
  ===   VIDEOIO
  ===   VIDEOSTAB
  ===   VIZ
  ===   WECHAT_QRCODE
  ===   XFEATURES2D
  ===   XIMGPROC
  ===   XOBJDETECT
  ===   XPHOTO
  === Detected probe priority based on environment vars: pkg_config: false, cmake: false, vcpkg: false
  === Probing the OpenCV library in the following order: environment, pkg_config, cmake, vcpkg_cmake, vcpkg
  === Can't probe using: environment, continuing with other methods because: Some environment variables are missing
  === Probing OpenCV library using pkg_config
  === Successfully probed using: pkg_config
  === OpenCV library configuration: Library {
      include_paths: [
          "/usr/local/include/opencv4",
      ],
      version: Version {
          major: 4,
          minor: 8,
          patch: 0,
      },
      cargo_metadata: [
          "cargo:rustc-link-search=/usr/local/lib",
          "cargo:rustc-link-lib=opencv_gapi",
          "cargo:rustc-link-lib=opencv_highgui",
          "cargo:rustc-link-lib=opencv_ml",
          "cargo:rustc-link-lib=opencv_objdetect",
          "cargo:rustc-link-lib=opencv_photo",
          "cargo:rustc-link-lib=opencv_stitching",
          "cargo:rustc-link-lib=opencv_video",
          "cargo:rustc-link-lib=opencv_calib3d",
          "cargo:rustc-link-lib=opencv_features2d",
          "cargo:rustc-link-lib=opencv_dnn",
          "cargo:rustc-link-lib=opencv_flann",
          "cargo:rustc-link-lib=opencv_videoio",
          "cargo:rustc-link-lib=opencv_imgcodecs",
          "cargo:rustc-link-lib=opencv_imgproc",
          "cargo:rustc-link-lib=opencv_core",
      ],
  }

  thread 'main' panicked at /home/ahmadradhy/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/opencv-0.90.0/build.rs:355:10:
  Discovered OpenCV include paths is empty or contains non-existent paths
  note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
warning: build failed, waiting for other jobs to finish...

pkg-config --cflags opencv4
ls -ld /usr/include/opencv4 /usr/local/include/opencv4 2>/dev/null || true
pkg-config --libs opencv4


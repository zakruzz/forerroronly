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

ahmadradhy@ubuntu:~/tugasakhir-fairuz/yolo-trt-rs$ pkg-config --cflags opencv4
-I/usr/local/include/opencv4
ahmadradhy@ubuntu:~/tugasakhir-fairuz/yolo-trt-rs$ ls -ld /usr/include/opencv4 /usr/local/include/opencv4 2>/dev/null || true
drwxr-xr-x 3 root root 4096 Mar 17  2023 /usr/include/opencv4
ahmadradhy@ubuntu:~/tugasakhir-fairuz/yolo-trt-rs$ pkg-config --libs opencv4
-L/usr/local/lib -lopencv_gapi -lopencv_highgui -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_video -lopencv_calib3d -lopencv_features2d -lopencv_dnn -lopencv_flann -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_core


cat >/tmp/jetson_gpu_audit.sh <<'SH'
set -e
echo "== Jetson GPU Audit =="

echo "[CUDA]"
nvcc --version 2>/dev/null || true
[ -d /usr/local/cuda ] && echo "  ✓ /usr/local/cuda ada" || echo "  ✗ CUDA dir tak ditemukan"

echo "[TensorRT]"
which /usr/src/tensorrt/bin/trtexec 2>/dev/null && echo "  ✓ trtexec ada" || echo "  ✗ trtexec tak ditemukan"
dpkg -l | grep -E 'nvinfer|tensorrt' || echo "  (info) paket nvinfer/tensorrt tidak terdeteksi via dpkg list"

echo "[OpenCV dev]"
pkg-config --modversion opencv4 2>/dev/null && echo "  ✓ pkg-config opencv4 OK" || echo "  ✗ opencv4 via pkg-config tidak ada"
ls /usr/include/opencv4 1>/dev/null 2>&1 && echo "  ✓ header /usr/include/opencv4" || echo "  ✗ header opencv4 tak ditemukan"

echo "[GStreamer (opsional untuk capture lebih stabil)]"
gst-launch-1.0 --version 2>/dev/null || echo "  (info) gstreamer-runtime tidak terdeteksi"

echo "[LD paths]"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "  Cek libnvinfer:"
ldconfig -p | grep -i nvinfer || echo "  (info) ldconfig tidak menemukan libnvinfer, pastikan JetPack/TensorRT sudah terpasang"

echo "== Selesai =="
SH
bash /tmp/jetson_gpu_audit.sh

[10/05/2025-10:46:31] [E] Unknown option: --workspace 2048
[10/05/2025-10:46:31] [E] Unknown option: --buildOnly 
[10/05/2025-10:46:31] [E] Unknown option: --explicitBatch 
&&&& FAILED TensorRT.trtexec [TensorRT v100300] # /usr/src/tensorrt/bin/trtexec --onnx=./models/best.onnx --saveEngine=models/best_fp16.engine --explicitBatch --fp16 --workspace=2048 --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640 --buildOnly


/usr/src/tensorrt/bin/trtexec \
  --onnx=./models/best.onnx \
  --saveEngine=models/best_fp16.engine \
  --explicitBatch \
  --fp16 \
  --workspace=2048 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640 \
  --buildOnly

/usr/src/tensorrt/bin/trtexec \
  --onnx=./models/best.onnx \
  --saveEngine=./models/best_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:2048 \
  --skipInference

  /usr/src/tensorrt/bin/trtexec \
  --onnx=$(pwd)/models/best.onnx \
  --saveEngine=$(pwd)/models/best_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:2048 \
  --skipInference \
  --verbose 2>&1 | tee trtexec_build.log

[{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/config.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [0]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "0",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/config.rs"
		}
	},
	"severity": 8,
	"message": "expected a pattern, found a function call\nfunction calls are not allowed in patterns: <https://doc.rust-lang.org/book/ch19-00-patterns.html>",
	"source": "rustc",
	"startLineNumber": 46,
	"startColumn": 16,
	"endLineNumber": 46,
	"endColumn": 18,
	"relatedInformation": [
		{
			"startLineNumber": 1,
			"startColumn": 1,
			"endLineNumber": 1,
			"endColumn": 1,
			"message": "consider importing one of these tuple variants instead: `use std::result::Result::Ok;\n`, `use opencv::core::MatExprResult::Ok;\n`",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/config.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [0]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "0",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs"
		}
	},
	"severity": 8,
	"message": "the `?` operator can only be applied to values that implement `Try`\nthe trait `Try` is not implemented for `bool`",
	"source": "rustc",
	"startLineNumber": 47,
	"startColumn": 38,
	"endLineNumber": 47,
	"endColumn": 52,
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [1]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "1",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs"
		}
	},
	"severity": 8,
	"message": "mismatched types\n  expected enum `std::result::Result<(), anyhow::Error>`\nfound unit type `()`",
	"source": "rustc",
	"startLineNumber": 48,
	"startColumn": 42,
	"endLineNumber": 48,
	"endColumn": 47,
	"relatedInformation": [
		{
			"startLineNumber": 31,
			"startColumn": 14,
			"endLineNumber": 31,
			"endColumn": 24,
			"message": "expected `Result<()>` because of this return type",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs"
		},
		{
			"startLineNumber": 46,
			"startColumn": 5,
			"endLineNumber": 46,
			"endColumn": 9,
			"message": "this loop is expected to be of type `std::result::Result<(), anyhow::Error>`",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs"
		},
		{
			"startLineNumber": 48,
			"startColumn": 47,
			"endLineNumber": 48,
			"endColumn": 47,
			"message": "give the `break` a value of the expected type: ` Ok(())`",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/preprocess.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [0]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "0",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/preprocess.rs"
		}
	},
	"severity": 8,
	"message": "cannot find function `copy_make_border` in module `imgproc`\nnot found in `imgproc`",
	"source": "rustc",
	"startLineNumber": 23,
	"startColumn": 14,
	"endLineNumber": 23,
	"endColumn": 30,
	"relatedInformation": [
		{
			"startLineNumber": 1,
			"startColumn": 1,
			"endLineNumber": 1,
			"endColumn": 1,
			"message": "consider importing one of these functions: `use crate::preprocess::core::copy_make_border;\n`, `use opencv::core::copy_make_border;\n`",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/preprocess.rs"
		},
		{
			"startLineNumber": 23,
			"startColumn": 5,
			"endLineNumber": 23,
			"endColumn": 14,
			"message": "if you import `copy_make_border`, refer to it directly",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/preprocess.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/preprocess.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [3]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "3",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/preprocess.rs"
		}
	},
	"severity": 8,
	"message": "cannot find value `BORDER_CONSTANT` in module `imgproc`\nnot found in `imgproc`",
	"source": "rustc",
	"startLineNumber": 25,
	"startColumn": 18,
	"endLineNumber": 25,
	"endColumn": 33,
	"relatedInformation": [
		{
			"startLineNumber": 1,
			"startColumn": 1,
			"endLineNumber": 1,
			"endColumn": 1,
			"message": "consider importing one of these items: `use crate::preprocess::core::BORDER_CONSTANT;\n`, `use crate::preprocess::core::BorderTypes::BORDER_CONSTANT;\n`, `use opencv::core::BORDER_CONSTANT;\n`, `use opencv::core::BorderTypes::BORDER_CONSTANT;\n`",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/preprocess.rs"
		},
		{
			"startLineNumber": 25,
			"startColumn": 9,
			"endLineNumber": 25,
			"endColumn": 18,
			"message": "if you import `BORDER_CONSTANT`, refer to it directly",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/preprocess.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/preprocess.rs",
	"owner": "rust-analyzer",
	"code": {
		"value": "E0605",
		"target": {
			"$mid": 1,
			"path": "/stable/error_codes/E0605.html",
			"scheme": "https",
			"authority": "doc.rust-lang.org"
		}
	},
	"severity": 8,
	"message": "non-primitive cast: `Result<*const u8, Error>` as `*const f32`",
	"source": "rust-analyzer",
	"startLineNumber": 41,
	"startColumn": 36,
	"endLineNumber": 41,
	"endColumn": 63,
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/preprocess.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [6]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "6",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/preprocess.rs"
		}
	},
	"severity": 8,
	"message": "non-primitive cast: `std::result::Result<*const u8, opencv::Error>` as `*const f32`\nan `as` expression can only be used to convert between primitive types or to coerce to a specific trait object",
	"source": "rustc",
	"startLineNumber": 41,
	"startColumn": 36,
	"endLineNumber": 41,
	"endColumn": 63,
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [0]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "0",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "unresolved imports `async_tensorrt::logger`, `async_tensorrt::context`, `async_tensorrt::cuda`\ncould not find `logger` in `async_tensorrt`",
	"source": "rustc",
	"startLineNumber": 6,
	"startColumn": 5,
	"endLineNumber": 6,
	"endColumn": 11,
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [1]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "1",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "unresolved imports `async_tensorrt::logger`, `async_tensorrt::context`, `async_tensorrt::cuda`\ncould not find `context` in `async_tensorrt`",
	"source": "rustc",
	"startLineNumber": 9,
	"startColumn": 5,
	"endLineNumber": 9,
	"endColumn": 12,
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [2]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "2",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "unresolved imports `async_tensorrt::logger`, `async_tensorrt::context`, `async_tensorrt::cuda`\ncould not find `cuda` in `async_tensorrt`",
	"source": "rustc",
	"startLineNumber": 10,
	"startColumn": 5,
	"endLineNumber": 10,
	"endColumn": 9,
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [3]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "3",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "this function takes 0 arguments but 1 argument was supplied",
	"source": "rustc",
	"startLineNumber": 33,
	"startColumn": 27,
	"endLineNumber": 33,
	"endColumn": 39,
	"relatedInformation": [
		{
			"startLineNumber": 33,
			"startColumn": 40,
			"endLineNumber": 33,
			"endColumn": 47,
			"message": "unexpected argument",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		},
		{
			"startLineNumber": 18,
			"startColumn": 18,
			"endLineNumber": 18,
			"endColumn": 21,
			"message": "associated function defined here",
			"resource": "/home/ahmadradhy/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/async-tensorrt-0.9.1/src/runtime.rs"
		},
		{
			"startLineNumber": 33,
			"startColumn": 40,
			"endLineNumber": 33,
			"endColumn": 47,
			"message": "remove the extra argument",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [6]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "6",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "the `?` operator can only be applied to values that implement `Try`\nthe trait `Try` is not implemented for `async_tensorrt::Runtime`",
	"source": "rustc",
	"startLineNumber": 33,
	"startColumn": 27,
	"endLineNumber": 33,
	"endColumn": 55,
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rust-analyzer",
	"code": {
		"value": "E0107",
		"target": {
			"$mid": 1,
			"path": "/stable/error_codes/E0107.html",
			"scheme": "https",
			"authority": "doc.rust-lang.org"
		}
	},
	"severity": 8,
	"message": "expected 0 arguments, found 1",
	"source": "rust-analyzer",
	"startLineNumber": 33,
	"startColumn": 39,
	"endLineNumber": 33,
	"endColumn": 48,
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [7]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "7",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "type annotations needed",
	"source": "rustc",
	"startLineNumber": 36,
	"startColumn": 17,
	"endLineNumber": 36,
	"endColumn": 23,
	"relatedInformation": [
		{
			"startLineNumber": 38,
			"startColumn": 38,
			"endLineNumber": 38,
			"endColumn": 62,
			"message": "type must be known at this point",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		},
		{
			"startLineNumber": 36,
			"startColumn": 23,
			"endLineNumber": 36,
			"endColumn": 23,
			"message": "consider giving `engine` an explicit type: `: /* Type */`",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [10]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "10",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "function takes 1 generic argument but 2 generic arguments were supplied\nexpected 1 generic argument",
	"source": "rustc",
	"startLineNumber": 62,
	"startColumn": 13,
	"endLineNumber": 62,
	"endColumn": 15,
	"relatedInformation": [
		{
			"startLineNumber": 653,
			"startColumn": 8,
			"endLineNumber": 653,
			"endColumn": 10,
			"message": "function defined here, with 1 generic parameter: `T`",
			"resource": "/home/ahmadradhy/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anyhow-1.0.100/src/lib.rs"
		},
		{
			"startLineNumber": 62,
			"startColumn": 19,
			"endLineNumber": 62,
			"endColumn": 34,
			"message": "remove the unnecessary generic argument",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [12]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "12",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "function takes 1 generic argument but 2 generic arguments were supplied\nexpected 1 generic argument",
	"source": "rustc",
	"startLineNumber": 100,
	"startColumn": 13,
	"endLineNumber": 100,
	"endColumn": 15,
	"relatedInformation": [
		{
			"startLineNumber": 653,
			"startColumn": 8,
			"endLineNumber": 653,
			"endColumn": 10,
			"message": "function defined here, with 1 generic parameter: `T`",
			"resource": "/home/ahmadradhy/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anyhow-1.0.100/src/lib.rs"
		},
		{
			"startLineNumber": 100,
			"startColumn": 19,
			"endLineNumber": 100,
			"endColumn": 34,
			"message": "remove the unnecessary generic argument",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [14]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "14",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "no method named `get_tensor_shape` found for struct `Arc<async_tensorrt::Engine>` in the current scope",
	"source": "rustc",
	"startLineNumber": 104,
	"startColumn": 49,
	"endLineNumber": 104,
	"endColumn": 65,
	"relatedInformation": [
		{
			"startLineNumber": 104,
			"startColumn": 49,
			"endLineNumber": 104,
			"endColumn": 65,
			"message": "there is a method `tensor_shape` with a similar name: `tensor_shape`",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/yolo.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [0]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "0",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/yolo.rs"
		}
	},
	"severity": 8,
	"message": "type annotations needed for `Vec<_>`",
	"source": "rustc",
	"startLineNumber": 24,
	"startColumn": 9,
	"endLineNumber": 24,
	"endColumn": 17,
	"relatedInformation": [
		{
			"startLineNumber": 27,
			"startColumn": 28,
			"endLineNumber": 27,
			"endColumn": 38,
			"message": "type must be known at this point",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/yolo.rs"
		},
		{
			"startLineNumber": 24,
			"startColumn": 17,
			"endLineNumber": 24,
			"endColumn": 17,
			"message": "consider giving `keep` an explicit type, where the type for type parameter `T` is specified: `: Vec<T>`",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/yolo.rs"
		}
	],
	"origin": "extHost1"
}]




[{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/config.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [0]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "0",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/config.rs"
		}
	},
	"severity": 8,
	"message": "expected a pattern, found a function call\nfunction calls are not allowed in patterns: <https://doc.rust-lang.org/book/ch19-00-patterns.html>",
	"source": "rustc",
	"startLineNumber": 46,
	"startColumn": 16,
	"endLineNumber": 46,
	"endColumn": 18,
	"relatedInformation": [
		{
			"startLineNumber": 1,
			"startColumn": 1,
			"endLineNumber": 1,
			"endColumn": 1,
			"message": "consider importing one of these tuple variants instead: `use std::result::Result::Ok;\n`, `use opencv::core::MatExprResult::Ok;\n`",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/config.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [0]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "0",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs"
		}
	},
	"severity": 8,
	"message": "expected a pattern, found a function call\nfunction calls are not allowed in patterns: <https://doc.rust-lang.org/book/ch19-00-patterns.html>",
	"source": "rustc",
	"startLineNumber": 42,
	"startColumn": 5,
	"endLineNumber": 42,
	"endColumn": 7,
	"relatedInformation": [
		{
			"startLineNumber": 6,
			"startColumn": 1,
			"endLineNumber": 6,
			"endColumn": 1,
			"message": "consider importing one of these tuple variants instead: `use std::result::Result::Ok;\n`, `use opencv::core::MatExprResult::Ok;\n`",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [2]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "2",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs"
		}
	},
	"severity": 8,
	"message": "the `?` operator can only be applied to values that implement `Try`\nthe trait `Try` is not implemented for `bool`",
	"source": "rustc",
	"startLineNumber": 56,
	"startColumn": 15,
	"endLineNumber": 56,
	"endColumn": 29,
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [0]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "0",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "unresolved imports `async_tensorrt::execution_context`, `async_tensorrt::memory`, `async_tensorrt::stream`\ncould not find `execution_context` in `async_tensorrt`",
	"source": "rustc",
	"startLineNumber": 8,
	"startColumn": 5,
	"endLineNumber": 8,
	"endColumn": 22,
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [1]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "1",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "unresolved imports `async_tensorrt::execution_context`, `async_tensorrt::memory`, `async_tensorrt::stream`\ncould not find `memory` in `async_tensorrt`",
	"source": "rustc",
	"startLineNumber": 9,
	"startColumn": 5,
	"endLineNumber": 9,
	"endColumn": 11,
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [2]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "2",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "unresolved imports `async_tensorrt::execution_context`, `async_tensorrt::memory`, `async_tensorrt::stream`\ncould not find `stream` in `async_tensorrt`",
	"source": "rustc",
	"startLineNumber": 10,
	"startColumn": 5,
	"endLineNumber": 10,
	"endColumn": 11,
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [5]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "5",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "type annotations needed",
	"source": "rustc",
	"startLineNumber": 34,
	"startColumn": 17,
	"endLineNumber": 34,
	"endColumn": 23,
	"relatedInformation": [
		{
			"startLineNumber": 36,
			"startColumn": 34,
			"endLineNumber": 36,
			"endColumn": 58,
			"message": "type must be known at this point",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		},
		{
			"startLineNumber": 34,
			"startColumn": 23,
			"endLineNumber": 34,
			"endColumn": 23,
			"message": "consider giving `engine` an explicit type: `: /* Type */`",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [3]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "3",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "no method named `deserialize_engine` found for opaque type `impl Future<Output = async_tensorrt::Runtime>` in the current scope\nmethod not found in `impl Future<Output = async_tensorrt::Runtime>`",
	"source": "rustc",
	"startLineNumber": 34,
	"startColumn": 34,
	"endLineNumber": 34,
	"endColumn": 52,
	"relatedInformation": [
		{
			"startLineNumber": 34,
			"startColumn": 34,
			"endLineNumber": 34,
			"endColumn": 34,
			"message": "consider `await`ing on the `Future` and calling the method on its `Output`: `await.`",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [8]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "8",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "function takes 1 generic argument but 2 generic arguments were supplied\nexpected 1 generic argument",
	"source": "rustc",
	"startLineNumber": 59,
	"startColumn": 13,
	"endLineNumber": 59,
	"endColumn": 15,
	"relatedInformation": [
		{
			"startLineNumber": 653,
			"startColumn": 8,
			"endLineNumber": 653,
			"endColumn": 10,
			"message": "function defined here, with 1 generic parameter: `T`",
			"resource": "/home/ahmadradhy/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anyhow-1.0.100/src/lib.rs"
		},
		{
			"startLineNumber": 59,
			"startColumn": 19,
			"endLineNumber": 59,
			"endColumn": 34,
			"message": "remove the unnecessary generic argument",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [10]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "10",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	},
	"severity": 8,
	"message": "function takes 1 generic argument but 2 generic arguments were supplied\nexpected 1 generic argument",
	"source": "rustc",
	"startLineNumber": 91,
	"startColumn": 13,
	"endLineNumber": 91,
	"endColumn": 15,
	"relatedInformation": [
		{
			"startLineNumber": 653,
			"startColumn": 8,
			"endLineNumber": 653,
			"endColumn": 10,
			"message": "function defined here, with 1 generic parameter: `T`",
			"resource": "/home/ahmadradhy/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anyhow-1.0.100/src/lib.rs"
		},
		{
			"startLineNumber": 91,
			"startColumn": 19,
			"endLineNumber": 91,
			"endColumn": 34,
			"message": "remove the unnecessary generic argument",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/trt.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/yolo.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [0]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "0",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/yolo.rs"
		}
	},
	"severity": 8,
	"message": "cannot find value `a_y1` in this scope\nnot found in this scope",
	"source": "rustc",
	"startLineNumber": 17,
	"startColumn": 43,
	"endLineNumber": 17,
	"endColumn": 47,
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/yolo.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [1]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "1",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/yolo.rs"
		}
	},
	"severity": 8,
	"message": "cannot find value `b_y1` in this scope\nnot found in this scope",
	"source": "rustc",
	"startLineNumber": 18,
	"startColumn": 43,
	"endLineNumber": 18,
	"endColumn": 47,
	"origin": "extHost1"
}]

warning: unused import: `MatTraitManual`
 --> src/main.rs:3:72
  |
3 | ...Const, MatTraitConstManual, MatTraitManual, Size, Vec3b},
  |                                ^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

error[E0599]: no method named `context` found for struct `async_tensorrt::Runtime` in the current scope
   --> src/main.rs:131:35
    |
131 |     let rt = Runtime::new().await.context("create TRT runtime")?;
    |                                   ^^^^^^^ method not found in `async_tensorrt::Runtime`

error[E0599]: no function or associated item named `is_opened` found for struct `VideoCapture` in the current scope
    --> src/main.rs:42:32
     |
  42 |     if !videoio::VideoCapture::is_opened(&cap)? {
     |                                ^^^^^^^^^ function or associated item not found in `VideoCapture`
     |
note: if you're trying to build a new `VideoCapture` consider using one of the following associated functions:
      VideoCapture::default
      VideoCapture::from_file
      VideoCapture::from_file_def
      VideoCapture::from_file_with_params
      and 3 others
    --> /home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/target/debug/build/opencv-747386007b8b1744/out/opencv/videoio.rs:1210:3
     |
1210 |         pub fn default() -> Result<crate::videoio::VideoCapture> {
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1245 |         pub fn from_file(filename: &str, api_preference: i32) -> Result<crate::videoio::VideoCapture> {
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1282 |         pub fn from_file_def(filename: &str) -> Result<crate::videoio::VideoCapture> {
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
1305 |         pub fn from_file_with_params(filename: &str, api_preference: i32, params: &core::Vector<i32>) -> Result<crate::videoio::VideoCapture> {
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     = help: items from traits can only be used if the trait is in scope
help: trait `VideoCaptureTraitConst` which provides `is_opened` is implemented but not in scope; perhaps you want to import it
     |
   1 + use opencv::prelude::VideoCaptureTraitConst;
     |

error[E0599]: no method named `set` found for struct `VideoCapture` in the current scope
    --> src/main.rs:46:17
     |
  46 |     let _ = cap.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0);
     |                 ^^^
     |
    ::: /home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/target/debug/build/opencv-747386007b8b1744/out/opencv/videoio.rs:1793:6
     |
1793 |         fn set(&mut self, prop_id: i32, value: f64) -> Result<bool> {
     |            --- the method is available for `VideoCapture` here
     |
     = help: items from traits can only be used if the trait is in scope
help: there is a method `get` with a similar name, but with different arguments
    --> /home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/target/debug/build/opencv-747386007b8b1744/out/opencv/videoio.rs:1492:3
     |
1492 |         fn get(&self, prop_id: i32) -> Result<f64> {
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
help: trait `VideoCaptureTrait` which provides `set` is implemented but not in scope; perhaps you want to import it
     |
   1 + use opencv::prelude::VideoCaptureTrait;
     |

error[E0599]: no method named `set` found for struct `VideoCapture` in the current scope
    --> src/main.rs:47:17
     |
  47 |     let _ = cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0);
     |                 ^^^
     |
    ::: /home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/target/debug/build/opencv-747386007b8b1744/out/opencv/videoio.rs:1793:6
     |
1793 |         fn set(&mut self, prop_id: i32, value: f64) -> Result<bool> {
     |            --- the method is available for `VideoCapture` here
     |
     = help: items from traits can only be used if the trait is in scope
help: there is a method `get` with a similar name, but with different arguments
    --> /home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/target/debug/build/opencv-747386007b8b1744/out/opencv/videoio.rs:1492:3
     |
1492 |         fn get(&self, prop_id: i32) -> Result<f64> {
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
help: trait `VideoCaptureTrait` which provides `set` is implemented but not in scope; perhaps you want to import it
     |
   1 + use opencv::prelude::VideoCaptureTrait;
     |

error[E0599]: no method named `read` found for struct `VideoCapture` in the current scope
    --> src/main.rs:52:13
     |
  52 |         cap.read(&mut frame)?;
     |             ^^^^ method not found in `VideoCapture`
     |
    ::: /home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/target/debug/build/opencv-747386007b8b1744/out/opencv/videoio.rs:1772:6
     |
1772 |         fn read(&mut self, image: &mut impl ToOutputArray) -> Result<bool> {
     |            ---- the method is available for `VideoCapture` here
     |
     = help: items from traits can only be used if the trait is in scope
help: trait `VideoCaptureTrait` which provides `read` is implemented but not in scope; perhaps you want to import it
     |
   1 + use opencv::prelude::VideoCaptureTrait;
     |

error[E0061]: this function takes 2 arguments but 1 argument was supplied
  --> src/main.rs:62:22
   |
62 |         let in_dev = DeviceBuffer::from_slice(&input_tensor)
   |                      ^^^^^^^^^^^^^^^^^^^^^^^^--------------- argument #2 of type `&async_cuda::Stream` is missing
   |
note: associated function defined here
  --> /home/ahmadradhy/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/async-cuda-0.6.1/src/memory/device.rs:61:18
   |
61 |     pub async fn from_slice(slice: &[T], stream: &Stream) -> Result<Self> {
   |                  ^^^^^^^^^^
help: provide the argument
   |
62 |         let in_dev = DeviceBuffer::from_slice(&input_tensor, /* &async_cuda::Stream */)
   |                                                            +++++++++++++++++++++++++++

error[E0599]: no method named `context` found for opaque type `impl Future<Output = Result<..., ...>>` in the current scope
  --> src/main.rs:63:14
   |
62 |           let in_dev = DeviceBuffer::from_slice(&input_tensor)
   |  ______________________-
63 | |             .context("copy H->D input tensor")?;
   | |             -^^^^^^^ method not found in `impl Future<Output = Result<..., ...>>`
   | |_____________|
   |

error[E0061]: this function takes 2 arguments but 1 argument was supplied
  --> src/main.rs:68:27
   |
68 |         let mut out_dev = DeviceBuffer::from_slice(&out_host)
   |                           ^^^^^^^^^^^^^^^^^^^^^^^^----------- argument #2 of type `&async_cuda::Stream` is missing
   |
note: associated function defined here
  --> /home/ahmadradhy/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/async-cuda-0.6.1/src/memory/device.rs:61:18
   |
61 |     pub async fn from_slice(slice: &[T], stream: &Stream) -> Result<Self> {
   |                  ^^^^^^^^^^
help: provide the argument
   |
68 |         let mut out_dev = DeviceBuffer::from_slice(&out_host, /* &async_cuda::Stream */)
   |                                                             +++++++++++++++++++++++++++

error[E0599]: no method named `context` found for opaque type `impl Future<Output = Result<..., ...>>` in the current scope
  --> src/main.rs:69:14
   |
68 |           let mut out_dev = DeviceBuffer::from_slice(&out_host)
   |  ___________________________-
69 | |             .context("alloc D output buffer")?;
   | |             -^^^^^^^ method not found in `impl Future<Output = Result<..., ...>>`
   | |_____________|
   |

warning: unused import: `MatTraitConstManual`
 --> src/main.rs:3:51
  |
3 |     core::{self, BorderTypes, Mat, MatTraitConst, MatTraitConstManual, Ma...
  |                                                   ^^^^^^^^^^^^^^^^^^^

Some errors have detailed explanations: E0061, E0599.
For more information about an error, try `rustc --explain E0061`.
warning: `veh-counter-rs` (bin "veh-counter-rs") generated 2 warnings
error: could not compile `veh-counter-rs` (bin "veh-counter-rs") due to 9 previous errors; 2 warnings emitted


ahmadradhy@ubuntu:~/tugasakhir-fairuz/veh-counter-rs$ cargo build
   Compiling veh-counter-rs v0.1.0 (/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs)
warning: unused import: `path::Path`
 --> src/main.rs:9:37
  |
9 | use std::{collections::HashMap, fs, path::Path};
  |                                     ^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

error[E0599]: no method named `create_execution_context` found for struct `async_tensorrt::Engine` in the current scope
  --> src/main.rs:31:10
   |
30 |       let mut ctx = engine
   |  ___________________-
31 | |         .create_execution_context()
   | |         -^^^^^^^^^^^^^^^^^^^^^^^^ method not found in `async_tensorrt::Engine`
   | |_________|
   |

error[E0308]: mismatched types
  --> src/main.rs:67:64
   |
67 |         let mut d_in = DeviceBuffer::from_slice(&input_tensor, &stream)
   |                        ------------------------                ^^^^^^^ expected `&Stream`, found `&Result<Stream, Error>`
   |                        |
   |                        arguments to this function are incorrect
   |
   = note: expected reference `&async_cuda::Stream`
              found reference `&std::result::Result<async_cuda::Stream, async_cuda::Error>`
note: associated function defined here
  --> /home/ahmadradhy/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/async-cuda-0.6.1/src/memory/device.rs:61:18
   |
61 |     pub async fn from_slice(slice: &[T], stream: &Stream) -> Result<Self> {
   |                  ^^^^^^^^^^

error[E0308]: mismatched types
  --> src/main.rs:71:61
   |
71 |         let mut d_out = DeviceBuffer::<f32>::new(out_elems, &stream)
   |                         ------------------------            ^^^^^^^ expected `&Stream`, found `&Result<Stream, Error>`
   |                         |
   |                         arguments to this function are incorrect
   |
   = note: expected reference `&async_cuda::Stream`
              found reference `&std::result::Result<async_cuda::Stream, async_cuda::Error>`
note: associated function defined here
  --> /home/ahmadradhy/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/async-cuda-0.6.1/src/memory/device.rs:42:18
   |
42 |     pub async fn new(num_elements: usize, stream: &Stream) -> Self {
   |                  ^^^

error[E0599]: no method named `context` found for struct `async_cuda::DeviceBuffer` in the current scope
  --> src/main.rs:73:14
   |
71 |           let mut d_out = DeviceBuffer::<f32>::new(out_elems, &stream)
   |  _________________________-
72 | |             .await
73 | |             .context("alloc D output buffer")?;
   | |             -^^^^^^^ method not found in `async_cuda::DeviceBuffer<f32>`
   | |_____________|
   |

Some errors have detailed explanations: E0308, E0599.
For more information about an error, try `rustc --explain E0308`.
warning: `veh-counter-rs` (bin "veh-counter-rs") generated 1 warning
error: could not compile `veh-counter-rs` (bin "veh-counter-rs") due to 4 previous errors; 1 warning emitted





ahmadradhy@ubuntu:~/tugasakhir-fairuz/veh-counter-rs$ cargo build
   Compiling veh-counter-rs v0.1.0 (/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs)
error[E0432]: unresolved import `async_tensorrt::execution_context`
  --> src/main.rs:14:5
   |
14 |     execution_context::ExecutionContext,
   |     ^^^^^^^^^^^^^^^^^ could not find `execution_context` in `async_tensorrt`

error[E0599]: no method named `clone` found for struct `async_tensorrt::Engine` in the current scope
  --> src/main.rs:37:56
   |
37 |     let mut ctx = ExecutionContext::from_engine(engine.clone())
   |                                                        ^^^^^ method not found in `async_tensorrt::Engine`

error[E0308]: mismatched types
   --> src/main.rs:89:22
    |
 89 |             .copy_to(&mut out_host, &stream)
    |              ------- ^^^^^^^^^^^^^ expected `&mut HostBuffer<f32>`, found `&mut Vec<f32>`
    |              |
    |              arguments to this method are incorrect
    |
    = note: expected mutable reference `&mut async_cuda::HostBuffer<f32>`
               found mutable reference `&mut Vec<f32>`
note: method defined here
   --> /home/ahmadradhy/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/async-cuda-0.6.1/src/memory/device.rs:174:18
    |
174 |     pub async fn copy_to(&self, other: &mut HostBuffer<T>, stream: &Str...
    |                  ^^^^^^^

Some errors have detailed explanations: E0308, E0432, E0599.
For more information about an error, try `rustc --explain E0308`.
error: could not compile `veh-counter-rs` (bin "veh-counter-rs") due to 3 previous errors



ahmadradhy@ubuntu:~/tugasakhir-fairuz/veh-counter-rs$ cargo build
   Compiling veh-counter-rs v0.1.0 (/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs)
warning: unused import: `self`
 --> src/main.rs:6:12
  |
6 |     core::{self, Mat, MatTraitConst, MatTrait, Size, Vec3b},
  |            ^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: unused import: `serde::Deserialize`
  --> src/main.rs:11:5
   |
11 | use serde::Deserialize;
   |     ^^^^^^^^^^^^^^^^^^

error[E0599]: no method named `deserialize_engine` found for opaque type `impl Future<Output = Runtime>` in the current scope
   --> src/main.rs:132:10
    |
131 |       let mut engine: Engine = rt
    |  ______________________________-
132 | |         .deserialize_engine(&plan)
    | |         -^^^^^^^^^^^^^^^^^^ method not found in `impl Future<Output = Runtime>`
    | |_________|
    |
    |
help: consider `await`ing on the `Future` and calling the method on its `Output`
    |
132 |         .await.deserialize_engine(&plan)
    |          ++++++

error[E0277]: the `?` operator can only be applied to values that implement `Try`
   --> src/main.rs:182:12
    |
182 |         if frame.empty()? {
    |            ^^^^^^^^^^^^^^ the `?` operator cannot be applied to type `bool`
    |
    = help: the trait `Try` is not implemented for `bool`

warning: unused import: `MatTrait`
 --> src/main.rs:6:38
  |
6 |     core::{self, Mat, MatTraitConst, MatTrait, Size, Vec3b},
  |                                      ^^^^^^^^

Some errors have detailed explanations: E0277, E0599.
For more information about an error, try `rustc --explain E0277`.
warning: `veh-counter-rs` (bin "veh-counter-rs") generated 3 warnings
error: could not compile `veh-counter-rs` (bin "veh-counter-rs") due to 2 previous errors; 3 warnings emitted




ahmadradhy@ubuntu:~/tugasakhir-fairuz/veh-counter-rs$ cargo build
   Compiling veh-counter-rs v0.1.0 (/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs)
warning: unused import: `prelude`
 --> src/main.rs:8:5
  |
8 |     prelude::*,
  |     ^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

error[E0004]: non-exhaustive patterns: `TensorIoMode::None` not covered
   --> src/main.rs:123:15
    |
123 |         match engine.tensor_io_mode(&name) {
    |               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ pattern `TensorIoMode::None` not covered
    |
note: `TensorIoMode` defined here
   --> /home/ahmadradhy/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/async-tensorrt-0.9.1/src/ffi/sync/engine.rs:326:1
    |
326 | pub enum TensorIoMode {
    | ^^^^^^^^^^^^^^^^^^^^^
327 |     None,
    |     ---- not covered
    = note: the matched value is of type `TensorIoMode`
help: ensure that all possible cases are being handled by adding a match arm with a wildcard pattern or an explicit pattern as shown
    |
125 ~             TensorIoMode::Output => if output_name.is_none() { output_name = Some(name); },
126 ~             TensorIoMode::None => todo!(),
    |

For more information about this error, try `rustc --explain E0004`.
warning: `veh-counter-rs` (bin "veh-counter-rs") generated 1 warning
error: could not compile `veh-counter-rs` (bin "veh-counter-rs") due to 1 previous error; 1 warning emitted





[{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs",
	"owner": "rustc",
	"code": {
		"value": "Click for full compiler diagnostic",
		"target": {
			"$mid": 1,
			"path": "/diagnostic message [0]",
			"scheme": "rust-analyzer-diagnostics-view",
			"query": "0",
			"fragment": "file:///home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs"
		}
	},
	"severity": 8,
	"message": "non-exhaustive patterns: `TensorIoMode::None` not covered\nthe matched value is of type `TensorIoMode`",
	"source": "rustc",
	"startLineNumber": 122,
	"startColumn": 15,
	"endLineNumber": 122,
	"endColumn": 43,
	"relatedInformation": [
		{
			"startLineNumber": 326,
			"startColumn": 1,
			"endLineNumber": 326,
			"endColumn": 22,
			"message": "`TensorIoMode` defined here",
			"resource": "/home/ahmadradhy/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/async-tensorrt-0.9.1/src/ffi/sync/engine.rs"
		},
		{
			"startLineNumber": 124,
			"startColumn": 91,
			"endLineNumber": 124,
			"endColumn": 91,
			"message": "ensure that all possible cases are being handled by adding a match arm with a wildcard pattern or an explicit pattern as shown: `,\n            TensorIoMode::None => todo!()`",
			"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs"
		}
	],
	"origin": "extHost1"
},{
	"resource": "/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs/src/main.rs",
	"owner": "rust-analyzer",
	"code": {
		"value": "E0004",
		"target": {
			"$mid": 1,
			"path": "/stable/error_codes/E0004.html",
			"scheme": "https",
			"authority": "doc.rust-lang.org"
		}
	},
	"severity": 8,
	"message": "missing match arm: `None` not covered",
	"source": "rust-analyzer",
	"startLineNumber": 122,
	"startColumn": 15,
	"endLineNumber": 122,
	"endColumn": 43,
	"origin": "extHost1"
}]


ahmadradhy@ubuntu:~/tugasakhir-fairuz/veh-counter-rs$ cargo build
   Compiling veh-counter-rs v0.1.0 (/home/ahmadradhy/tugasakhir-fairuz/veh-counter-rs)
error[E0282]: type annotations needed for `Vec<_>`
  --> src/main.rs:93:9
   |
93 |     let mut keep=Vec::new();
   |         ^^^^^^^^
94 |     'o: for d in dets.into_iter(){
95 |         for k in &keep{ if d.cls==k.cls && iou(&d,k)>iou_thr {continue '...
   |                                   ----- type must be known at this point
   |
help: consider giving `keep` an explicit type, where the type for type parameter `T` is specified
   |
93 |     let mut keep: Vec<T>=Vec::new();
   |                 ++++++++

warning: unused variable: `shp_s`
   --> src/main.rs:281:17
    |
281 |             let shp_s = engine.tensor_shape(&s); // [1,max_det,1]
    |                 ^^^^^ help: if this is intentional, prefix it with an underscore: `_shp_s`
    |
    = note: `#[warn(unused_variables)]` on by default

warning: unused variable: `shp_c`
   --> src/main.rs:282:17
    |
282 |             let shp_c = engine.tensor_shape(&c); // [1,max_det,1]
    |                 ^^^^^ help: if this is intentional, prefix it with an underscore: `_shp_c`

warning: unused variable: `fps_counter`
   --> src/main.rs:306:13
    |
306 |     let mut fps_counter = 0u32;
    |             ^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_fps_counter`

warning: unused variable: `fps_last`
   --> src/main.rs:307:13
    |
307 |     let mut fps_last = Instant::now();
    |             ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_fps_last`

warning: unused variable: `fps`
   --> src/main.rs:308:13
    |
308 |     let mut fps = 0.0f32;
    |             ^^^ help: if this is intentional, prefix it with an underscore: `_fps`

warning: variable does not need to be mutable
   --> src/main.rs:306:9
    |
306 |     let mut fps_counter = 0u32;
    |         ----^^^^^^^^^^^
    |         |
    |         help: remove this `mut`
    |
    = note: `#[warn(unused_mut)]` on by default

warning: variable does not need to be mutable
   --> src/main.rs:307:9
    |
307 |     let mut fps_last = Instant::now();
    |         ----^^^^^^^^
    |         |
    |         help: remove this `mut`

warning: variable does not need to be mutable
   --> src/main.rs:308:9
    |
308 |     let mut fps = 0.0f32;
    |         ----^^^
    |         |
    |         help: remove this `mut`

error[E0502]: cannot borrow `engine` as immutable because it is also borrowed as mutable
   --> src/main.rs:273:25
    |
252 |     let mut ctx = ExecutionContext::new(&mut engine).await.context("cre...
    |                                         ----------- mutable borrow occurs here
...
273 |         let out_shape = engine.tensor_shape(&outputs[0]);
    |                         ^^^^^^ immutable borrow occurs here
...
419 | }
    | - mutable borrow might be used here, when `ctx` is dropped and runs the destructor for type `async_tensorrt::ExecutionContext<'_>`

error[E0502]: cannot borrow `engine` as immutable because it is also borrowed as mutable
   --> src/main.rs:279:58
    |
252 |     let mut ctx = ExecutionContext::new(&mut engine).await.context("cre...
    |                                         ----------- mutable borrow occurs here
...
279 |         if let Some((b,s,c)) = pick_efficientnms_outputs(&engine, &outp...
    |                                                          ^^^^^^^ immutable borrow occurs here
...
419 | }
    | - mutable borrow might be used here, when `ctx` is dropped and runs the destructor for type `async_tensorrt::ExecutionContext<'_>`

error[E0502]: cannot borrow `engine` as immutable because it is also borrowed as mutable
   --> src/main.rs:280:25
    |
252 |     let mut ctx = ExecutionContext::new(&mut engine).await.context("cre...
    |                                         ----------- mutable borrow occurs here
...
280 |             let shp_b = engine.tensor_shape(&b); // [1,max_det,4] biasanya
    |                         ^^^^^^ immutable borrow occurs here
...
419 | }
    | - mutable borrow might be used here, when `ctx` is dropped and runs the destructor for type `async_tensorrt::ExecutionContext<'_>`

error[E0502]: cannot borrow `engine` as immutable because it is also borrowed as mutable
   --> src/main.rs:281:25
    |
252 |     let mut ctx = ExecutionContext::new(&mut engine).await.context("cre...
    |                                         ----------- mutable borrow occurs here
...
281 |             let shp_s = engine.tensor_shape(&s); // [1,max_det,1]
    |                         ^^^^^^ immutable borrow occurs here
...
419 | }
    | - mutable borrow might be used here, when `ctx` is dropped and runs the destructor for type `async_tensorrt::ExecutionContext<'_>`

error[E0502]: cannot borrow `engine` as immutable because it is also borrowed as mutable
   --> src/main.rs:282:25
    |
252 |     let mut ctx = ExecutionContext::new(&mut engine).await.context("cre...
    |                                         ----------- mutable borrow occurs here
...
282 |             let shp_c = engine.tensor_shape(&c); // [1,max_det,1]
    |                         ^^^^^^ immutable borrow occurs here
...
419 | }
    | - mutable borrow might be used here, when `ctx` is dropped and runs the destructor for type `async_tensorrt::ExecutionContext<'_>`

error[E0502]: cannot borrow `engine` as immutable because it is also borrowed as mutable
   --> src/main.rs:296:29
    |
252 |     let mut ctx = ExecutionContext::new(&mut engine).await.context("cre...
    |                                         ----------- mutable borrow occurs here
...
296 |             let out_shape = engine.tensor_shape(&outputs[0]);
    |                             ^^^^^^ immutable borrow occurs here
...
419 | }
    | - mutable borrow might be used here, when `ctx` is dropped and runs the destructor for type `async_tensorrt::ExecutionContext<'_>`

error[E0502]: cannot borrow `engine` as immutable because it is also borrowed as mutable
   --> src/main.rs:329:53
    |
252 |     let mut ctx = ExecutionContext::new(&mut engine).await.context("cre...
    |                                         ----------- mutable borrow occurs here
...
329 |             let (b,s,c) = pick_efficientnms_outputs(&engine, &outputs)....
    |                                                     ^^^^^^^ immutable borrow occurs here
...
336 |         ctx.enqueue(&mut io_map, &stream).await.context("enqueue")?;
    |         --- mutable borrow later used here

error[E0597]: `b` does not live long enough
   --> src/main.rs:330:27
    |
329 | ...       let (b,s,c) = pick_efficientnms_outputs(&engine, &outputs).un...
    |                - binding `b` declared here
330 | ...       io_map.insert(&b, d_boxes.as_mut().unwrap());
    |                         ^^ borrowed value does not live long enough
...
333 | ...   }
    |       - `b` dropped here while still borrowed
...
336 | ...   ctx.enqueue(&mut io_map, &stream).await.context("enqueue")?;
    |                   ----------- borrow later used here

error[E0597]: `s` does not live long enough
   --> src/main.rs:331:27
    |
329 | ...       let (b,s,c) = pick_efficientnms_outputs(&engine, &outputs).un...
    |                  - binding `s` declared here
330 | ...       io_map.insert(&b, d_boxes.as_mut().unwrap());
331 | ...       io_map.insert(&s, d_scores.as_mut().unwrap());
    |                         ^^ borrowed value does not live long enough
332 | ...       io_map.insert(&c, d_classes.as_mut().unwrap());
333 | ...   }
    |       - `s` dropped here while still borrowed
...
336 | ...   ctx.enqueue(&mut io_map, &stream).await.context("enqueue")?;
    |                   ----------- borrow later used here

error[E0597]: `c` does not live long enough
   --> src/main.rs:332:27
    |
329 | ...       let (b,s,c) = pick_efficientnms_outputs(&engine, &outputs).un...
    |                    - binding `c` declared here
...
332 | ...       io_map.insert(&c, d_classes.as_mut().unwrap());
    |                         ^^ borrowed value does not live long enough
333 | ...   }
    |       - `c` dropped here while still borrowed
...
336 | ...   ctx.enqueue(&mut io_map, &stream).await.context("enqueue")?;
    |                   ----------- borrow later used here

error[E0502]: cannot borrow `engine` as immutable because it is also borrowed as mutable
   --> src/main.rs:343:29
    |
252 |     let mut ctx = ExecutionContext::new(&mut engine).await.context("cre...
    |                                         ----------- mutable borrow occurs here
...
343 |             let out_shape = engine.tensor_shape(&outputs[0]);
    |                             ^^^^^^ immutable borrow occurs here
...
419 | }
    | - mutable borrow might be used here, when `ctx` is dropped and runs the destructor for type `async_tensorrt::ExecutionContext<'_>`

Some errors have detailed explanations: E0282, E0502, E0597.
For more information about an error, try `rustc --explain E0282`.
warning: `veh-counter-rs` (bin "veh-counter-rs") generated 8 warnings
error: could not compile `veh-counter-rs` (bin "veh-counter-rs") due to 12 previous errors; 8 warnings emitted




/usr/src/tensorrt/bin/trtexec \
  --loadEngine=models/best_fp16.engine \
  --shapes=images:1x3x640x640 \
  --useCudaGraph \
  --streams=2 \
  --warmUp=10 \
  --duration=20 \
  --iterations=0 \
  --avgRuns=100 \
  --dumpProfile \
  --verbose


  /usr/src/tensorrt/bin/trtexec \
  --onnx=models/best.onnx \
  --saveEngine=models/best_fp16.engine \
  --shapes=images:1x3x640x640 \
  --fp16 --useCudaGraph \
  --memPoolSize=workspace:2048 \
  --noTF32


# contoh untuk ONNX fixed 640×640
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/best.onnx \
  --saveEngine=models/best_fp16.engine \
  --fp16 \
  --useCudaGraph \
  --memPoolSize=workspace:2048 \
  --noTF32


/usr/src/tensorrt/bin/trtexec \
  --onnx=best.onnx \
  --saveEngine=yolo_fp16.plan \
  --fp16 \
  --workspace=4096 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640



[10/05/2025-15:32:55] [E] Static model does not take explicit shapes since the shape of inference tensors will be determined by the model itself
[10/05/2025-15:32:55] [E] Network And Config setup failed
[10/05/2025-15:32:55] [E] Building engine failed
[10/05/2025-15:32:55] [E] Failed to create engine from model or file.
[10/05/2025-15:32:55] [E] Engine set up failed
&&&& FAILED TensorRT.trtexec [TensorRT v100300] # /usr/src/tensorrt/bin/trtexec --onnx=models/best.onnx --saveEngine=models/best_fp16.engine --shapes=images:1x3x640x640 --fp16 --useCudaGraph --memPoolSize=workspace:2048 --noTF32


# CUDA, cuDNN, TensorRT
nvcc --version
dpkg -l | grep -E 'TensorRT|nvinfer|nvonnxparser|cudnn|cuda'

# GStreamer dev & plugin
gst-inspect-1.0 | head

# Rust toolchain
rustc --version || true





ahmadradhy@ubuntu:~/Image_processing$ cargo run -- ./best.onnx ./classes.json ./jalan2.jpeg 640 0.25 0.45  result1.jpg
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.06s
     Running `target/debug/image_processor ./best.onnx ./classes.json ./jalan2.jpeg 640 0.25 0.45 result1.jpg`
[info] DNN backend: CUDA
[ WARN:0@0.069] global net_impl.cpp:178 setUpNet DNN module was not built with CUDA backend; switching to CPU
vehicle_count: 0
Error: OpenCV(4.8.0) /home/ubuntu/opencv_build/opencv/modules/imgcodecs/src/loadsave.cpp:696: error: (-2:Unspecified error) could not find a writer for the specified extension in function 'imwrite_'
 (code: StsError, -2)


"""Repository rule for CUDA autoconfiguration.

`cuda_configure` depends on the following environment variables:

  * `NEED_CUDA`: Whether to enable building with CUDA.
  * `GCC_HOST_COMPILER_PATH`: The GCC host compiler path
  * `SYSROOT`: The sysroot to use when compiling.
  * `CUDA_PATHS`: The base paths to look for CUDA and cuDNN. Default is
    `/usr/local/cuda,usr/`.
  * `CUDA_TOOLKIT_PATH` (deprecated): The path to the CUDA toolkit. Default is
    `/usr/local/cuda`.
  * `CUDA_VERSION`: The version of the CUDA toolkit. If this is blank, then
    use the system default.
  * `CUDNN_VERSION`: The version of the cuDNN library.
  * `CUDNN_INSTALL_PATH` (deprecated): The path to the cuDNN library. Default is
    `/usr/local/cuda`.
  * `CUDA_COMPUTE_CAPABILITIES`: The CUDA compute capabilities. Default is
    `3.5,5.2`.
  * `PYTHON_BIN_PATH`: The python binary path
"""

load(
    "@bazel_tools//tools/cpp:lib_cc_configure.bzl",
    "escape_string",
    "get_env_var",
)
load(
    "//build_deps/remote_config:common.bzl",
    "config_repo_label",
    "err_out",
    "execute",
    "get_bash_bin",
    "get_cpu_value",
    "get_host_environ",
    "get_python_bin",
    "raw_exec",
    "read_dir",
    "realpath",
    "which",
)

_GCC_HOST_COMPILER_PATH = "GCC_HOST_COMPILER_PATH"
_GCC_HOST_COMPILER_PREFIX = "GCC_HOST_COMPILER_PREFIX"
_SYSROOT = "SYSROOT"
_CUDA_TOOLKIT_PATH = "CUDA_TOOLKIT_PATH"
_CUDA_VERSION = "CUDA_VERSION"
_CUDNN_VERSION = "CUDNN_VERSION"
_CUDNN_INSTALL_PATH = "CUDNN_INSTALL_PATH"
_CUDA_COMPUTE_CAPABILITIES = "CUDA_COMPUTE_CAPABILITIES"
_CUDA_CONFIG_REPO = "CUDA_CONFIG_REPO"
_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"

_TENSORRT_VERSION = "TENSORRT_VERSION"
_TENSORRT_INSTALL_PATH = "TENSORRT_INSTALL_PATH"
_TENSORRT_STATIC_PATH = "TENSORRT_STATIC_PATH"
_TENSORRT_LIBS = [
    "nvinfer",
    "nvinfer_plugin",
    "nvonnxparser",
    "nvparsers",
]
_TENSORRT_HEADERS = [
    "NvInfer.h",
    "NvUtils.h",
    "NvInferPlugin.h",
]
_TENSORRT_HEADERS_V6 = [
    "NvInfer.h",
    "NvUtils.h",
    "NvInferPlugin.h",
    "NvInferVersion.h",
    "NvInferRuntime.h",
    "NvInferRuntimeCommon.h",
    "NvInferPluginUtils.h",
    "NvOnnxParser.h",
    "NvOnnxConfig.h",
]
_TENSORRT_HEADERS_V8 = [
    "NvInfer.h",
    "NvInferLegacyDims.h",
    "NvInferImpl.h",
    "NvUtils.h",
    "NvInferPlugin.h",
    "NvInferVersion.h",
    "NvInferRuntime.h",
    "NvInferRuntimeCommon.h",
    "NvInferPluginUtils.h",
    "NvOnnxParser.h",
    "NvOnnxConfig.h",
]

def _at_least_version(actual_version, required_version):
    actual = [int(v) for v in actual_version.split(".")]
    required = [int(v) for v in required_version.split(".")]
    return actual >= required

def _get_tensorrt_headers(tensorrt_version):
    if _at_least_version(tensorrt_version, "8"):
        return _TENSORRT_HEADERS_V8
    if _at_least_version(tensorrt_version, "6"):
        return _TENSORRT_HEADERS_V6
    return _TENSORRT_HEADERS

def to_list_of_strings(elements):
    """Convert the list of ["a", "b", "c"] into '"a", "b", "c"'.

    This is to be used to put a list of strings into the bzl file templates
    so it gets interpreted as list of strings in Starlark.

    Args:
      elements: list of string elements

    Returns:
      single string of elements wrapped in quotes separated by a comma."""
    quoted_strings = ["\"" + element + "\"" for element in elements]
    return ", ".join(quoted_strings)

def verify_build_defines(params):
    """Verify all variables that crosstool/BUILD.tpl expects are substituted.

    Args:
      params: dict of variables that will be passed to the BUILD.tpl template.
    """
    missing = []
    for param in [
        "cxx_builtin_include_directories",
        "extra_no_canonical_prefixes_flags",
        "host_compiler_path",
        "host_compiler_prefix",
        "host_compiler_warnings",
        "linker_bin_path",
        "compiler_deps",
        "unfiltered_compile_flags",
    ]:
        if ("%{" + param + "}") not in params:
            missing.append(param)

    if missing:
        auto_configure_fail(
            "BUILD.tpl template is missing these variables: " + str(missing) +
            ".\nWe only got: " + str(params) + ".",
        )

# TODO(dzc): Once these functions have been factored out of Bazel's
# cc_configure.bzl, load them from @bazel_tools instead.
# BEGIN cc_configure common functions.
def find_cc(repository_ctx):
    """Find the C++ compiler."""
    target_cc_name = "gcc"
    cc_path_envvar = _GCC_HOST_COMPILER_PATH
    cc_name = target_cc_name

    cc_name_from_env = get_host_environ(repository_ctx, cc_path_envvar)
    if cc_name_from_env:
        cc_name = cc_name_from_env
    if cc_name.startswith("/"):
        # Absolute path, maybe we should make this supported by our which function.
        return cc_name
    cc = which(repository_ctx, cc_name)
    if cc == None:
        fail(("Cannot find {}, either correct your path or set the {}" +
              " environment variable").format(target_cc_name, cc_path_envvar))
    return cc

_INC_DIR_MARKER_BEGIN = "#include <...>"

# OSX add " (framework directory)" at the end of line, strip it.
_OSX_FRAMEWORK_SUFFIX = " (framework directory)"
_OSX_FRAMEWORK_SUFFIX_LEN = len(_OSX_FRAMEWORK_SUFFIX)

def _cxx_inc_convert(path):
    """Convert path returned by cc -E xc++ in a complete path."""
    path = path.strip()
    if path.endswith(_OSX_FRAMEWORK_SUFFIX):
        path = path[:-_OSX_FRAMEWORK_SUFFIX_LEN].strip()
    return path

def _normalize_include_path(repository_ctx, path):
    """Normalizes include paths before writing them to the crosstool.

      If path points inside the 'crosstool' folder of the repository, a relative
      path is returned.
      If path points outside the 'crosstool' folder, an absolute path is returned.
      """
    path = str(repository_ctx.path(path))
    crosstool_folder = str(repository_ctx.path(".").get_child("crosstool"))

    if path.startswith(crosstool_folder):
        # We drop the path to "$REPO/crosstool" and a trailing path separator.
        return path[len(crosstool_folder) + 1:]
    return path

def _is_compiler_option_supported(repository_ctx, cc, option):
    """Checks that `option` is supported by the C compiler. Doesn't %-escape the option."""
    result = repository_ctx.execute([
        cc,
        option,
        "-o",
        "/dev/null",
        "-c",
        str(repository_ctx.path("tools/cpp/empty.cc")),
    ])
    return result.stderr.find(option) == -1

def _get_cxx_inc_directories_impl(repository_ctx, cc, lang_is_cpp, tf_sysroot):
    """Compute the list of default C or C++ include directories."""
    if lang_is_cpp:
        lang = "c++"
    else:
        lang = "c"
    sysroot = []
    if tf_sysroot:
        sysroot += ["--sysroot", tf_sysroot]
    result = raw_exec(
        repository_ctx,
        [cc, "-E", "-x" + lang, "-", "-v"] + sysroot,
    )
    stderr = err_out(result)
    index1 = stderr.find(_INC_DIR_MARKER_BEGIN)
    if index1 == -1:
        return []
    index1 = stderr.find("\n", index1)
    if index1 == -1:
        return []
    index2 = stderr.rfind("\n ")
    if index2 == -1 or index2 < index1:
        return []
    index2 = stderr.find("\n", index2 + 1)
    if index2 == -1:
        inc_dirs = stderr[index1 + 1:]
    else:
        inc_dirs = stderr[index1 + 1:index2].strip()

    print_resource_dir_supported = _is_compiler_option_supported(
        repository_ctx,
        cc,
        "-print-resource-dir",
    )

    if print_resource_dir_supported:
        resource_dir = repository_ctx.execute(
            [cc, "-print-resource-dir"],
        ).stdout.strip() + "/share"
        inc_dirs += "\n" + resource_dir

    return [
        _normalize_include_path(repository_ctx, _cxx_inc_convert(p))
        for p in inc_dirs.split("\n")
    ]

def get_cxx_inc_directories(repository_ctx, cc, tf_sysroot):
    """Compute the list of default C and C++ include directories."""

    includes_cpp = _get_cxx_inc_directories_impl(
        repository_ctx,
        cc,
        True,
        tf_sysroot,
    )
    includes_c = _get_cxx_inc_directories_impl(
        repository_ctx,
        cc,
        False,
        tf_sysroot,
    )

    return includes_cpp + [
        inc
        for inc in includes_c
        if inc not in includes_cpp
    ]

def auto_configure_fail(msg):
    """Output failure message when cuda configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sCuda Configuration Error:%s %s\n" % (red, no_color, msg))

# END cc_configure common functions (see TODO above).

def _cuda_include_path(repository_ctx, cuda_config):
    """Generates the Starlark string with cuda include directories.

      Args:
        repository_ctx: The repository context.
        cc: The path to the gcc host compiler.

      Returns:
        A list of the gcc host compiler include directories.
      """
    nvcc_path = repository_ctx.path("%s/bin/nvcc%s" % (
        cuda_config.cuda_toolkit_path,
        ".exe" if cuda_config.cpu_value == "Windows" else "",
    ))

    # The expected exit code of this command is non-zero. Bazel remote execution
    # only caches commands with zero exit code. So force a zero exit code.
    cmd = "%s -v /dev/null -o /dev/null ; [ $? -eq 1 ]" % str(nvcc_path)
    result = raw_exec(
        repository_ctx,
        [get_bash_bin(repository_ctx), "-c", cmd],
    )
    target_dir = ""
    for one_line in err_out(result).splitlines():
        if one_line.startswith("#$ _TARGET_DIR_="):
            target_dir = (cuda_config.cuda_toolkit_path + "/" +
                          one_line.replace(
                              "#$ _TARGET_DIR_=",
                              "",
                          ) + "/include")
    inc_entries = []
    if target_dir != "":
        inc_entries.append(realpath(repository_ctx, target_dir))
    inc_entries.append(
        realpath(repository_ctx, cuda_config.cuda_toolkit_path + "/include"),
    )
    return inc_entries

def matches_version(environ_version, detected_version):
    """Checks whether the user-specified version matches the detected version.

      This function performs a weak matching so that if the user specifies only
      the
      major or major and minor versions, the versions are still considered
      matching
      if the version parts match. To illustrate:

          environ_version  detected_version  result
          -----------------------------------------
          5.1.3            5.1.3             True
          5.1              5.1.3             True
          5                5.1               True
          5.1.3            5.1               False
          5.2.3            5.1.3             False

      Args:
        environ_version: The version specified by the user via environment
          variables.
        detected_version: The version autodetected from the CUDA installation on
          the system.
      Returns: True if user-specified version matches detected version and False
        otherwise.
    """
    environ_version_parts = environ_version.split(".")
    detected_version_parts = detected_version.split(".")
    if len(detected_version_parts) < len(environ_version_parts):
        return False
    for i, part in enumerate(detected_version_parts):
        if i >= len(environ_version_parts):
            break
        if part != environ_version_parts[i]:
            return False
    return True

_NVCC_VERSION_PREFIX = "Cuda compilation tools, release "

_DEFINE_CUDNN_MAJOR = "#define CUDNN_MAJOR"

def compute_capabilities(repository_ctx):
    """Returns a list of strings representing cuda compute capabilities.

    Args:
      repository_ctx: the repo rule's context.
    Returns: list of cuda architectures to compile for. 'compute_xy' refers to
      both PTX and SASS, 'sm_xy' refers to SASS only.
    """
    capabilities = get_host_environ(
        repository_ctx,
        _CUDA_COMPUTE_CAPABILITIES,
        "compute_35,compute_52",
    ).split(",")

    # Map old 'x.y' capabilities to 'compute_xy'.
    if len(capabilities) > 0 and all(
        [len(x.split(".")) == 2 for x in capabilities],
    ):
        # If all capabilities are in 'x.y' format, only include PTX for the
        # highest capability.
        cc_list = sorted([x.replace(".", "") for x in capabilities])
        capabilities = [
            "sm_%s" % x
            for x in cc_list[:-1]
        ] + ["compute_%s" % cc_list[-1]]
    for i, capability in enumerate(capabilities):
        parts = capability.split(".")
        if len(parts) != 2:
            continue
        capabilities[i] = "compute_%s%s" % (parts[0], parts[1])

    # Make list unique
    capabilities = dict(zip(capabilities, capabilities)).keys()

    # Validate capabilities.
    for capability in capabilities:
        if not capability.startswith(("compute_", "sm_")):
            auto_configure_fail("Invalid compute capability: %s" % capability)
        for prefix in ["compute_", "sm_"]:
            if not capability.startswith(prefix):
                continue
            if len(capability) == len(prefix) + 2 and capability[-2:].isdigit(
            ):
                continue
            auto_configure_fail("Invalid compute capability: %s" % capability)

    return capabilities

def lib_name(base_name, cpu_value, version = None, static = False):
    """Constructs the platform-specific name of a library.

      Args:
        base_name: The name of the library, such as "cudart"
        cpu_value: The name of the host operating system.
        version: The version of the library.
        static: True the library is static or False if it is a shared object.

      Returns:
        The platform-specific name of the library.
      """
    version = "" if not version else "." + version
    if cpu_value in ("Linux", "FreeBSD"):
        if static:
            return "lib%s.a" % base_name
        return "lib%s.so%s" % (base_name, version)
    elif cpu_value == "Windows":
        return "%s.lib" % base_name
    elif cpu_value == "Darwin":
        if static:
            return "lib%s.a" % base_name
        return "lib%s%s.dylib" % (base_name, version)
    else:
        auto_configure_fail("Invalid cpu_value: %s" % cpu_value)

def _lib_path(lib, cpu_value, basedir, version, static):
    file_name = lib_name(lib, cpu_value, version, static)
    return "%s/%s" % (basedir, file_name)

def _should_check_soname(version, static):
    return version and not static

def _check_cuda_lib_params(lib, cpu_value, basedir, version, static = False):
    return (
        _lib_path(lib, cpu_value, basedir, version, static),
        _should_check_soname(version, static),
    )

def _check_cuda_libs(repository_ctx, script_path, libs):
    python_bin = get_python_bin(repository_ctx)
    contents = repository_ctx.read(script_path).splitlines()

    cmd = "from os import linesep;"
    cmd += "f = open('script.py', 'w');"
    for line in contents:
        cmd += "f.write('%s' + linesep);" % line
    cmd += "f.close();"
    cmd += "from os import system;"
    args = " ".join(["\"" + path + "\" " + str(check) for path, check in libs])
    cmd += "system('%s script.py %s');" % (python_bin, args)

    all_paths = [path for path, _ in libs]
    checked_paths = execute(
        repository_ctx,
        [python_bin, "-c", cmd],
    ).stdout.splitlines()

    # Filter out empty lines from splitting on '\r\n' on Windows
    checked_paths = [path for path in checked_paths if len(path) > 0]
    if all_paths != checked_paths:
        auto_configure_fail(
            "Error with installed CUDA libs. Expected '%s'. Actual '%s'." %
            (all_paths, checked_paths),
        )

def _find_libs(repository_ctx, check_cuda_libs_script, cuda_config):
    """Returns the CUDA and cuDNN libraries on the system.

      Also, verifies that the script actually exist.

      Args:
        repository_ctx: The repository context.
        check_cuda_libs_script: The path to a script verifying that the cuda
          libraries exist on the system.
        cuda_config: The CUDA config as returned by _get_cuda_config

      Returns:
        Map of library names to structs of filename and path.
      """
    cpu_value = cuda_config.cpu_value
    stub_dir = "/stubs"

    check_cuda_libs_params = {
        "cuda": _check_cuda_lib_params(
            "cuda",
            cpu_value,
            cuda_config.config["cuda_library_dir"] + stub_dir,
            version = None,
            static = False,
        ),
        "cudart": _check_cuda_lib_params(
            "cudart",
            cpu_value,
            cuda_config.config["cuda_library_dir"],
            cuda_config.cudart_version,
            static = False,
        ),
        "cudart_static": _check_cuda_lib_params(
            "cudart_static",
            cpu_value,
            cuda_config.config["cuda_library_dir"],
            cuda_config.cudart_version,
            static = True,
        ),
        "cublas": _check_cuda_lib_params(
            "cublas",
            cpu_value,
            cuda_config.config["cublas_library_dir"],
            cuda_config.cublas_version,
            static = False,
        ),
        "cublasLt": _check_cuda_lib_params(
            "cublasLt",
            cpu_value,
            cuda_config.config["cublas_library_dir"],
            cuda_config.cublas_version,
            static = False,
        ),
        "cusolver": _check_cuda_lib_params(
            "cusolver",
            cpu_value,
            cuda_config.config["cusolver_library_dir"],
            cuda_config.cusolver_version,
            static = False,
        ),
        "curand": _check_cuda_lib_params(
            "curand",
            cpu_value,
            cuda_config.config["curand_library_dir"],
            cuda_config.curand_version,
            static = False,
        ),
        "cufft": _check_cuda_lib_params(
            "cufft",
            cpu_value,
            cuda_config.config["cufft_library_dir"],
            cuda_config.cufft_version,
            static = False,
        ),
        "cudnn": _check_cuda_lib_params(
            "cudnn",
            cpu_value,
            cuda_config.config["cudnn_library_dir"],
            cuda_config.cudnn_version,
            static = False,
        ),
        "cupti": _check_cuda_lib_params(
            "cupti",
            cpu_value,
            cuda_config.config["cupti_library_dir"],
            cuda_config.cupti_version,
            static = False,
        ),
        "cusparse": _check_cuda_lib_params(
            "cusparse",
            cpu_value,
            cuda_config.config["cusparse_library_dir"],
            cuda_config.cusparse_version,
            static = False,
        ),
    }

    # Verify that the libs actually exist at their locations.
    _check_cuda_libs(
        repository_ctx,
        check_cuda_libs_script,
        check_cuda_libs_params.values(),
    )

    paths = {
        filename: v[0]
        for (filename, v) in check_cuda_libs_params.items()
    }
    return paths

def _cudart_static_linkopt(cpu_value):
    """Returns additional platform-specific linkopts for cudart."""
    return "" if cpu_value == "Darwin" else "\"-lrt\","

def _exec_find_cuda_config(repository_ctx, script_path, cuda_libraries):
    python_bin = get_python_bin(repository_ctx)
    cmd = "from os import system;" + "system('\"%s\" %s %s');" % (
        python_bin,
        script_path,
        " ".join(cuda_libraries),
    )
    return execute(repository_ctx, [python_bin, "-c", cmd])

# TODO(csigg): Only call once instead of from here, tensorrt_configure.bzl,
# and nccl_configure.bzl.
def find_cuda_config(repository_ctx, script_path, cuda_libraries):
    """Returns CUDA config dictionary from running find_cuda_config.py"""
    exec_result = _exec_find_cuda_config(
        repository_ctx,
        script_path,
        cuda_libraries,
    )

    if exec_result.return_code:
        auto_configure_fail("Failed to run find_cuda_config.py: %s" %
                            err_out(exec_result))

    # Parse the dict from stdout.
    return dict(
        [tuple(x.split(": ")) for x in exec_result.stdout.splitlines()],
    )

def _get_cuda_config(repository_ctx, find_cuda_config_script):
    """Detects and returns information about the CUDA installation on the system.

      Args:
        repository_ctx: The repository context.

      Returns:
        A struct containing the following fields:
          cuda_toolkit_path: The CUDA toolkit installation directory.
          cudnn_install_basedir: The cuDNN installation directory.
          cuda_version: The version of CUDA on the system.
          cudart_version: The CUDA runtime version on the system.
          cudnn_version: The version of cuDNN on the system.
          compute_capabilities: A list of the system's CUDA compute capabilities.
          cpu_value: The name of the host operating system.
      """
    config = find_cuda_config(
        repository_ctx,
        find_cuda_config_script,
        ["cuda", "cudnn"],
    )

    cpu_value = get_cpu_value(repository_ctx)
    toolkit_path = config["cuda_toolkit_path"]

    cuda_version = config["cuda_version"].split(".")
    cuda_major = cuda_version[0]
    cuda_minor = cuda_version[1]

    cuda_version = "%s.%s" % (cuda_major, cuda_minor)
    cudnn_version = "%s" % config["cudnn_version"]

    if int(cuda_major) >= 11:
        # The libcudart soname in CUDA 11.x is versioned as 11.0 for backward compatability.
        if int(cuda_major) == 11:
            cudart_version = "11.0"
            cupti_version = cuda_version
        else:
            cudart_version = ("%s") % cuda_major
            cupti_version = cudart_version
        cublas_version = ("%s") % config["cublas_version"].split(".")[0]
        cusolver_version = ("%s") % config["cusolver_version"].split(".")[0]
        curand_version = ("%s") % config["curand_version"].split(".")[0]
        cufft_version = ("%s") % config["cufft_version"].split(".")[0]
        cusparse_version = ("%s") % config["cusparse_version"].split(".")[0]
    elif (int(cuda_major), int(cuda_minor)) >= (10, 1):
        # cuda_lib_version is for libraries like cuBLAS, cuFFT, cuSOLVER, etc.
        # It changed from 'x.y' to just 'x' in CUDA 10.1.
        cuda_lib_version = ("%s") % cuda_major
        cudart_version = cuda_version
        cupti_version = cuda_version
        cublas_version = cuda_lib_version
        cusolver_version = cuda_lib_version
        curand_version = cuda_lib_version
        cufft_version = cuda_lib_version
        cusparse_version = cuda_lib_version
    else:
        cudart_version = cuda_version
        cupti_version = cuda_version
        cublas_version = cuda_version
        cusolver_version = cuda_version
        curand_version = cuda_version
        cufft_version = cuda_version
        cusparse_version = cuda_version

    return struct(
        cuda_toolkit_path = toolkit_path,
        cuda_version = cuda_version,
        cupti_version = cupti_version,
        cuda_version_major = cuda_major,
        cudart_version = cudart_version,
        cublas_version = cublas_version,
        cusolver_version = cusolver_version,
        curand_version = curand_version,
        cufft_version = cufft_version,
        cusparse_version = cusparse_version,
        cudnn_version = cudnn_version,
        compute_capabilities = compute_capabilities(repository_ctx),
        cpu_value = cpu_value,
        config = config,
    )

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl.replace(":", "/")
    repository_ctx.template(
        out,
        Label("//build_deps/gpus/%s.tpl" % tpl),
        substitutions,
    )

def _file(repository_ctx, label):
    repository_ctx.template(
        label.replace(":", "/"),
        Label("//build_deps/gpus/%s.tpl" % label),
        {},
    )

_DUMMY_CROSSTOOL_BZL_FILE = """
def error_gpu_disabled():
  fail("ERROR: Building with --config=cuda but TensorFlow is not configured " +
       "to build with GPU support. Please re-run ./configure and enter 'Y' " +
       "at the prompt to build with GPU support.")

  native.genrule(
      name = "error_gen_crosstool",
      outs = ["CROSSTOOL"],
      cmd = "echo 'Should not be run.' && exit 1",
  )

  native.filegroup(
      name = "crosstool",
      srcs = [":CROSSTOOL"],
      output_licenses = ["unencumbered"],
  )
"""

_DUMMY_CROSSTOOL_BUILD_FILE = """
load("//crosstool:error_gpu_disabled.bzl", "error_gpu_disabled")

error_gpu_disabled()
"""

def _norm_path(path):
    """Returns a path with '/' and remove the trailing slash."""
    path = path.replace("\\", "/")
    if path[-1] == "/":
        path = path[:-1]
    return path

def make_copy_files_rule(repository_ctx, name, srcs, outs):
    """Returns a rule to copy a set of files."""
    cmds = []

    # Copy files.
    for src, out in zip(srcs, outs):
        cmds.append('cp -f "%s" "$(location %s)"' % (src, out))
    outs = [('        "%s",' % out) for out in outs]
    return """genrule(
    name = "%s",
    outs = [
%s
    ],
    cmd = \"""%s \""",
)""" % (name, "\n".join(outs), " && \\\n".join(cmds))

def make_copy_dir_rule(
        repository_ctx,
        name,
        src_dir,
        out_dir,
        exceptions = None):
    """Returns a rule to recursively copy a directory.
    If exceptions is not None, it must be a list of files or directories in
    'src_dir'; these will be excluded from copying.
    """
    src_dir = _norm_path(src_dir)
    out_dir = _norm_path(out_dir)
    outs = read_dir(repository_ctx, src_dir)
    post_cmd = ""
    if exceptions != None:
        outs = [
            x
            for x in outs
            if not any([x.startswith(src_dir + "/" + y) for y in exceptions])
        ]
    outs = [('        "%s",' % out.replace(src_dir, out_dir)) for out in outs]

    # '@D' already contains the relative path for a single file, see
    # http://docs.bazel.build/versions/master/be/make-variables.html#predefined_genrule_variables
    out_dir = "$(@D)/%s" % out_dir if len(outs) > 1 else "$(@D)"
    if exceptions != None:
        for x in exceptions:
            post_cmd += " ; rm -fR " + out_dir + "/" + x
    return """genrule(
    name = "%s",
    outs = [
%s
    ],
    cmd = \"""cp -rLf "%s/." "%s/" %s\""",
)""" % (name, "\n".join(outs), src_dir, out_dir, post_cmd)

def _flag_enabled(repository_ctx, flag_name):
    return get_host_environ(repository_ctx, flag_name) == "1"

def _tf_sysroot(repository_ctx):
    return get_host_environ(repository_ctx, _SYSROOT, "")

def _compute_cuda_extra_copts(repository_ctx, compute_capabilities):
    copts = []
    for capability in compute_capabilities:
        if capability.startswith("compute_"):
            capability = capability.replace("compute_", "sm_")
            copts.append("--cuda-include-ptx=%s" % capability)
        copts.append("--cuda-gpu-arch=%s" % capability)

    return str(copts)

def _tpl_path(repository_ctx, filename):
    return repository_ctx.path(Label("//build_deps/gpus/%s.tpl" % filename))

def _basename(repository_ctx, path_str):
    """Returns the basename of a path of type string.
    """

    num_chars = len(path_str)
    for i in range(num_chars):
        r_i = num_chars - 1 - i
        if path_str[r_i] == "/":
            return path_str[r_i + 1:]
    return path_str

def _create_local_cuda_repository(repository_ctx):
    """Creates the repository containing files set up to build with CUDA."""
    tpl_paths = {
        filename: _tpl_path(repository_ctx, filename)
        for filename in [
            "cuda:build_defs.bzl",
            "crosstool:crosstool_compiler_wrapper",
            "crosstool:BUILD",
            "crosstool:cc_toolchain_config.bzl",
            "cuda:cuda_config.h",
            "cuda:cuda_config.py",
        ]
    }
    tpl_paths["cuda:BUILD"] = _tpl_path(repository_ctx, "cuda:BUILD")
    find_cuda_config_script = repository_ctx.path(
        Label("//build_deps/gpus:find_cuda_config.py"),
    )

    cuda_config = _get_cuda_config(repository_ctx, find_cuda_config_script)

    cuda_include_path = cuda_config.config["cuda_include_dir"]
    cublas_include_path = cuda_config.config["cublas_include_dir"]
    cudnn_header_dir = cuda_config.config["cudnn_include_dir"]
    cupti_header_dir = cuda_config.config["cupti_include_dir"]
    nvvm_libdevice_dir = cuda_config.config["nvvm_library_dir"]

    # Create genrule to copy files from the installed CUDA toolkit into execroot.
    copy_rules = [
        make_copy_dir_rule(
            repository_ctx,
            name = "cuda-include",
            src_dir = cuda_include_path,
            out_dir = "cuda/include",
        ),
        make_copy_dir_rule(
            repository_ctx,
            name = "cuda-nvvm",
            src_dir = nvvm_libdevice_dir,
            out_dir = "cuda/nvvm/libdevice",
        ),
        make_copy_dir_rule(
            repository_ctx,
            name = "cuda-extras",
            src_dir = cupti_header_dir,
            out_dir = "cuda/extras/CUPTI/include",
        ),
    ]

    copy_rules.append(
        make_copy_files_rule(
            repository_ctx,
            name = "cublas-include",
            srcs = [
                cublas_include_path + "/cublas.h",
                cublas_include_path + "/cublas_v2.h",
                cublas_include_path + "/cublas_api.h",
                cublas_include_path + "/cublasLt.h",
            ],
            outs = [
                "cublas/include/cublas.h",
                "cublas/include/cublas_v2.h",
                "cublas/include/cublas_api.h",
                "cublas/include/cublasLt.h",
            ],
        ),
    )

    cusolver_include_path = cuda_config.config["cusolver_include_dir"]
    copy_rules.append(
        make_copy_files_rule(
            repository_ctx,
            name = "cusolver-include",
            srcs = [
                cusolver_include_path + "/cusolver_common.h",
                cusolver_include_path + "/cusolverDn.h",
            ],
            outs = [
                "cusolver/include/cusolver_common.h",
                "cusolver/include/cusolverDn.h",
            ],
        ),
    )

    cufft_include_path = cuda_config.config["cufft_include_dir"]
    copy_rules.append(
        make_copy_files_rule(
            repository_ctx,
            name = "cufft-include",
            srcs = [
                cufft_include_path + "/cufft.h",
            ],
            outs = [
                "cufft/include/cufft.h",
            ],
        ),
    )

    cusparse_include_path = cuda_config.config["cusparse_include_dir"]
    copy_rules.append(
        make_copy_files_rule(
            repository_ctx,
            name = "cusparse-include",
            srcs = [
                cusparse_include_path + "/cusparse.h",
            ],
            outs = [
                "cusparse/include/cusparse.h",
            ],
        ),
    )

    curand_include_path = cuda_config.config["curand_include_dir"]
    copy_rules.append(
        make_copy_files_rule(
            repository_ctx,
            name = "curand-include",
            srcs = [
                curand_include_path + "/curand.h",
            ],
            outs = [
                "curand/include/curand.h",
            ],
        ),
    )

    check_cuda_libs_script = repository_ctx.path(
        Label("//build_deps/gpus:check_cuda_libs.py"),
    )
    cuda_libs = _find_libs(repository_ctx, check_cuda_libs_script, cuda_config)
    cuda_lib_srcs = []
    cuda_lib_outs = []
    for path in cuda_libs.values():
        cuda_lib_srcs.append(path)
        cuda_lib_outs.append("cuda/lib/" + _basename(repository_ctx, path))
    copy_rules.append(
        make_copy_files_rule(
            repository_ctx,
            name = "cuda-lib",
            srcs = cuda_lib_srcs,
            outs = cuda_lib_outs,
        ),
    )

    file_ext = ""
    bin_files = (
        ["crt/link.stub"] +
        [f + file_ext for f in ["bin2c", "fatbinary", "nvlink", "nvprune"]]
    )
    copy_rules.append(
        make_copy_files_rule(
            repository_ctx,
            name = "cuda-bin",
            srcs = [
                cuda_config.cuda_toolkit_path + "/bin/" + f
                for f in bin_files
            ],
            outs = ["cuda/bin/" + f for f in bin_files],
        ),
    )

    # Select the headers based on the cuDNN version (strip '64_' for Windows).
    cudnn_headers = ["cudnn.h"]
    if cuda_config.cudnn_version.rsplit("_", 1)[-1] >= "8":
        cudnn_headers += [
            "cudnn_backend.h",
            "cudnn_adv_infer.h",
            "cudnn_adv_train.h",
            "cudnn_cnn_infer.h",
            "cudnn_cnn_train.h",
            "cudnn_ops_infer.h",
            "cudnn_ops_train.h",
            "cudnn_version.h",
        ]

    cudnn_srcs = []
    cudnn_outs = []
    for header in cudnn_headers:
        cudnn_srcs.append(cudnn_header_dir + "/" + header)
        cudnn_outs.append("cudnn/include/" + header)

    copy_rules.append(
        make_copy_files_rule(
            repository_ctx,
            name = "cudnn-include",
            srcs = cudnn_srcs,
            outs = cudnn_outs,
        ),
    )

    # Set up BUILD file for cuda/
    repository_ctx.template(
        "cuda/build_defs.bzl",
        tpl_paths["cuda:build_defs.bzl"],
        {
            "%{cuda_is_configured}": "True",
            "%{cuda_extra_copts}": _compute_cuda_extra_copts(
                repository_ctx,
                cuda_config.compute_capabilities,
            ),
            "%{cuda_gpu_architectures}": str(cuda_config.compute_capabilities),
        },
    )

    cub_actual = "@cub_archive//:cub"
    if int(cuda_config.cuda_version_major) >= 11:
        cub_actual = ":cuda_headers"

    repository_ctx.template(
        "cuda/BUILD",
        tpl_paths["cuda:BUILD"],
        {
            "%{cuda_driver_lib}": _basename(repository_ctx, cuda_libs["cuda"]),
            "%{cudart_static_lib}": _basename(repository_ctx, cuda_libs["cudart_static"]),
            "%{cudart_static_linkopt}": _cudart_static_linkopt(cuda_config.cpu_value),
            "%{cudart_lib}": _basename(repository_ctx, cuda_libs["cudart"]),
            "%{cublas_lib}": _basename(repository_ctx, cuda_libs["cublas"]),
            "%{cublasLt_lib}": _basename(repository_ctx, cuda_libs["cublasLt"]),
            "%{cusolver_lib}": _basename(repository_ctx, cuda_libs["cusolver"]),
            "%{cudnn_lib}": _basename(repository_ctx, cuda_libs["cudnn"]),
            "%{cufft_lib}": _basename(repository_ctx, cuda_libs["cufft"]),
            "%{curand_lib}": _basename(repository_ctx, cuda_libs["curand"]),
            "%{cupti_lib}": _basename(repository_ctx, cuda_libs["cupti"]),
            "%{cusparse_lib}": _basename(repository_ctx, cuda_libs["cusparse"]),
            "%{cub_actual}": cub_actual,
            "%{copy_rules}": "\n".join(copy_rules),
        },
    )

    tf_sysroot = _tf_sysroot(repository_ctx)

    # Set up crosstool/
    cc = find_cc(repository_ctx)
    cc_fullpath = cc

    host_compiler_includes = get_cxx_inc_directories(
        repository_ctx,
        cc_fullpath,
        tf_sysroot,
    )
    cuda_defines = {}
    cuda_defines["%{builtin_sysroot}"] = tf_sysroot
    cuda_defines["%{cuda_toolkit_path}"] = ""
    cuda_defines["%{compiler}"] = "unknown"

    host_compiler_prefix = get_host_environ(
        repository_ctx,
        _GCC_HOST_COMPILER_PREFIX,
    )
    if not host_compiler_prefix:
        host_compiler_prefix = "/usr/bin"

    cuda_defines["%{host_compiler_prefix}"] = host_compiler_prefix
    cuda_defines["%{linker_bin_path}"] = host_compiler_prefix
    cuda_defines["%{extra_no_canonical_prefixes_flags}"] = ""
    cuda_defines["%{unfiltered_compile_flags}"] = ""

    cuda_defines["%{host_compiler_path}"] = "crosstool_compiler_wrapper"
    cuda_defines["%{host_compiler_warnings}"] = ""

    # nvcc has the system include paths built in and will automatically
    # search them; we cannot work around that, so we add the relevant cuda
    # system paths to the allowed compiler specific include paths.
    cuda_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(
        host_compiler_includes + _cuda_include_path(
            repository_ctx,
            cuda_config,
        ) + [cupti_header_dir, cudnn_header_dir],
    )

    # For gcc, do not canonicalize system header paths; some versions of gcc
    # pick the shortest possible path for system includes when creating the
    # .d file - given that includes that are prefixed with "../" multiple
    # time quickly grow longer than the root of the tree, this can lead to
    # bazel's header check failing.
    cuda_defines["%{extra_no_canonical_prefixes_flags}"] = "\"-fno-canonical-system-headers\""

    file_ext = ""
    nvcc_path = "%s/nvcc%s" % (cuda_config.config["cuda_binary_dir"], file_ext)
    cuda_defines["%{compiler_deps}"] = ":crosstool_compiler"

    wrapper_defines = {
        "%{cpu_compiler}": str(cc),
        "%{cuda_version}": cuda_config.cuda_version,
        "%{nvcc_path}": nvcc_path,
        "%{gcc_host_compiler_path}": str(cc),
    }
    repository_ctx.template(
        "crosstool/crosstool_compiler_wrapper",
        tpl_paths["crosstool:crosstool_compiler_wrapper"],
        wrapper_defines,
    )

    verify_build_defines(cuda_defines)

    # Only expand template variables in the BUILD file
    repository_ctx.template(
        "crosstool/BUILD",
        tpl_paths["crosstool:BUILD"],
        cuda_defines,
    )

    # No templating of cc_toolchain_config - use attributes and templatize the
    # BUILD file.
    repository_ctx.template(
        "crosstool/cc_toolchain_config.bzl",
        tpl_paths["crosstool:cc_toolchain_config.bzl"],
        {},
    )

    # Set up cuda_config.h
    repository_ctx.template(
        "cuda/cuda/cuda_config.h",
        tpl_paths["cuda:cuda_config.h"],
        {
            "%{cuda_version}": cuda_config.cuda_version,
            "%{cudart_version}": cuda_config.cudart_version,
            "%{cupti_version}": cuda_config.cupti_version,
            "%{cublas_version}": cuda_config.cublas_version,
            "%{cusolver_version}": cuda_config.cusolver_version,
            "%{curand_version}": cuda_config.curand_version,
            "%{cufft_version}": cuda_config.cufft_version,
            "%{cusparse_version}": cuda_config.cusparse_version,
            "%{cudnn_version}": cuda_config.cudnn_version,
            "%{cuda_toolkit_path}": cuda_config.cuda_toolkit_path,
            "%{cuda_compute_capabilities}": ", ".join(
                [cc.split("_")[1] for cc in cuda_config.compute_capabilities],
            ),
        },
    )

    # Set up cuda_config.py, which is used by gen_build_info to provide
    # static build environment info to the API
    repository_ctx.template(
        "cuda/cuda/cuda_config.py",
        tpl_paths["cuda:cuda_config.py"],
        _py_tmpl_dict({
            "cuda_version": cuda_config.cuda_version,
            "cudnn_version": cuda_config.cudnn_version,
            "cuda_compute_capabilities": cuda_config.compute_capabilities,
            "cpu_compiler": str(cc),
        }),
    )

def _get_tensorrt_static_path(repository_ctx):
    return get_host_environ(repository_ctx, _TENSORRT_STATIC_PATH, None)

def _create_local_tensorrt_repository(repository_ctx):
    find_cuda_config_path = repository_ctx.path(
        Label("//build_deps/gpus:find_cuda_config.py"),
    )
    config = find_cuda_config(
        repository_ctx,
        find_cuda_config_path,
        ["tensorrt"],
    )
    tensorrt_version = config["tensorrt_version"]
    cpu_value = get_cpu_value(repository_ctx)

    # Copy the library and header files
    libraries = [
        lib_name(lib, cpu_value, tensorrt_version)
        for lib in _TENSORRT_LIBS
    ]
    library_dir = config["tensorrt_library_dir"] + "/"
    headers = _get_tensorrt_headers(tensorrt_version)
    include_dir = config["tensorrt_include_dir"] + "/"
    copy_rules = [
        make_copy_files_rule(
            repository_ctx,
            name = "tensorrt_lib",
            srcs = [library_dir + library for library in libraries],
            outs = ["tensorrt/lib/" + library for library in libraries],
        ),
        make_copy_files_rule(
            repository_ctx,
            name = "tensorrt_include",
            srcs = [include_dir + header for header in headers],
            outs = ["tensorrt/include/" + header for header in headers],
        ),
    ]

    tensorrt_static_path = _get_tensorrt_static_path(repository_ctx)
    if tensorrt_static_path:
        tensorrt_static_path = tensorrt_static_path + "/"
        if _at_least_version(tensorrt_version, "8"):
            raw_static_library_names = _TENSORRT_LIBS
        else:
            raw_static_library_names = _TENSORRT_LIBS + [
                "nvrtc",
                "myelin_compiler",
                "myelin_executor",
                "myelin_pattern_library",
                "myelin_pattern_runtime",
            ]

        static_library_names = [
            "%s_static" % name
            for name in raw_static_library_names
        ]
        static_libraries = [
            lib_name(lib, cpu_value, tensorrt_version, static = True)
            for lib in static_library_names
        ]
        copy_rules = copy_rules + [
            make_copy_files_rule(
                repository_ctx,
                name = "tensorrt_static_lib",
                srcs = [
                    tensorrt_static_path + library
                    for library in static_libraries
                ],
                outs = [
                    "tensorrt/lib/" + library
                    for library in static_libraries
                ],
            ),
        ]

    tpl_paths = {
        "tensorrt/build_defs.bzl": _tpl_path(repository_ctx, "tensorrt:build_defs.bzl"),
        "tensorrt/BUILD": _tpl_path(repository_ctx, "tensorrt:BUILD"),
        "tensorrt/tensorrt_config.h": _tpl_path(repository_ctx, "tensorrt:tensorrt_config.h"),
        "tensorrt/tensorrt_config.py": _tpl_path(repository_ctx, "tensorrt:tensorrt_config.py"),
    }

    # Set up config file.
    repository_ctx.template(
        "tensorrt/build_defs.bzl",
        tpl_paths["tensorrt/build_defs.bzl"],
        {"%{if_tensorrt}": "if_true"},
    )

    # Set up BUILD file.
    repository_ctx.template(
        "tensorrt/BUILD",
        tpl_paths["tensorrt/BUILD"],
        {
            "%{copy_rules}": "\n".join(copy_rules),
        },
    )

    # Set up tensorrt_config.h, which is used by
    # tensorflow/stream_executor/dso_loader.cc.
    repository_ctx.template(
        "tensorrt/tensorrt_config.h",
        tpl_paths["tensorrt/tensorrt_config.h"],
        {"%{tensorrt_version}": tensorrt_version},
    )

    # Set up tensorrt_config.py, which is used by gen_build_info to provide
    # build environment info to the API
    repository_ctx.template(
        "tensorrt/tensorrt_config.py",
        tpl_paths["tensorrt/tensorrt_config.py"],
        _py_tmpl_dict({
            "tensorrt_version": tensorrt_version,
        }),
    )

def _py_tmpl_dict(d):
    return {"%{cuda_config}": str(d)}

_CUDA_ENVIRONS = [
    _GCC_HOST_COMPILER_PATH,
    _GCC_HOST_COMPILER_PREFIX,
    "NEED_CUDA",
    _CUDA_TOOLKIT_PATH,
    _CUDNN_INSTALL_PATH,
    _CUDA_VERSION,
    _CUDNN_VERSION,
    _CUDA_COMPUTE_CAPABILITIES,
    "NVVMIR_LIBRARY_DIR",
    _PYTHON_BIN_PATH,
    "TMP",
    "TMPDIR",
    "CUDA_PATHS",
]

cuda_configure = repository_rule(
    implementation = _create_local_cuda_repository,
    environ = _CUDA_ENVIRONS,
)

_TENSORRT_ENVIRONS = [
    _TENSORRT_INSTALL_PATH,
    _TENSORRT_VERSION,
    _TENSORRT_STATIC_PATH,
    "CUDA_PATHS",
]

tensorrt_configure = repository_rule(
    implementation = _create_local_tensorrt_repository,
    environ = _TENSORRT_ENVIRONS,
)

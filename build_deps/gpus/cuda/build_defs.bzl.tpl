# Macros for building CUDA code.
def cuda_default_copts():
    """Default options for all CUDA compilations."""
    return [
        "-x",
        "cuda",
        "-DUSE_CUDA=1",
        "-Xcuda-fatbinary=--compress-all",
    ] + %{cuda_extra_copts}


def cuda_gpu_architectures():
    """Returns a list of supported GPU architectures."""
    return %{cuda_gpu_architectures}


def cuda_header_library(name,
                        hdrs,
                        include_prefix=None,
                        strip_include_prefix=None,
                        deps=[],
                        **kwargs):
    """Generates a cc_library containing both virtual and system include paths.

    Generates both a header-only target with virtual includes plus the full
    target without virtual includes. This works around the fact that bazel can't
    mix 'includes' and 'include_prefix' in the same target."""

    native.cc_library(
        name=name + "_virtual",
        hdrs=hdrs,
        include_prefix=include_prefix,
        strip_include_prefix=strip_include_prefix,
        deps=deps,
        visibility=["//visibility:private"],
    )

    native.cc_library(name=name,
                      textual_hdrs=hdrs,
                      deps=deps + [":%s_virtual" % name],
                      **kwargs)


def cuda_library(copts=[], **kwargs):
    """Wrapper over cc_library which adds default CUDA options."""
    native.cc_library(copts=cuda_default_copts() + copts, **kwargs)


def cuda_binary(copts=[], **kwargs):
    """Wrapper over cc_library which adds default CUDA options."""
    native.cc_binary(copts=cuda_default_copts() + copts, **kwargs)


def cuda_cc_test(copts=[], **kwargs):
    """Wrapper over cc_test which adds default CUDA options."""
    native.cc_test(copts=copts, **kwargs)

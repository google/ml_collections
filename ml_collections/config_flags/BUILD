# Description:
#   A library that allows complex configs flags.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "config_flags",
    srcs = [
        "__init__.py",
        "config_flags.py",
    ],
    srcs_version = "PY3",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":config_path",
        ":tuple_parser",
        # pip: absl/flags
        # pip: absl/logging
        "//ml_collections/config_dict",
    ],
)

py_library(
    name = "config_path",
    srcs = ["config_path.py"],
    deps = [
        "//ml_collections/config_dict",
    ],
)

py_library(
    name = "tuple_parser",
    srcs = [
        "tuple_parser.py",
    ],
    deps = [
        # pip: absl/flags
    ],
)

# Description:
#   A library that allows complex configs flags.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "config_flags",
    srcs = [
        "__init__.py",
        "config_flags.py",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":tuple_parser",
        # pip: absl/flags
        # pip: absl/logging
        "//ml_collections",
        # pip: six
        # pip: typing
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

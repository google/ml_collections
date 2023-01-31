# ML Collections is a library of Python Collections designed for ML usecases.

load("//tools/build_defs/license:license.bzl", "license")

package(
    default_applicable_licenses = ["//ml_collections:license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "ml_collections",
)

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "ml_collections",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//ml_collections/config_dict",
    ],
)

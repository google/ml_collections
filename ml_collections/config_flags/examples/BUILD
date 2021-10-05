# Package of examples for ml_collections.config_flags package.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_binary(
    name = "define_config_dict_basic",
    srcs = ["define_config_dict_basic.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//ml_collections/config_flags",
    ],
)

py_binary(
    name = "define_config_file_basic",
    srcs = ["define_config_file_basic.py"],
    data = [
        "config.py",
        "parameterised_config.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//ml_collections/config_flags",
    ],
)

py_library(
    name = "examples_lib",
    srcs = [
        "define_config_dict_basic.py",
        "define_config_file_basic.py",
    ],
    data = [
        "config.py",
        "parameterised_config.py",
    ],
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//ml_collections/config_flags",
    ],
)

py_test(
    name = "examples_test",
    srcs = ["examples_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":examples_lib",
        # pip: absl/flags
        # pip: absl/testing:absltest
        # pip: absl/testing:flagsaver
    ],
)

py_binary(
    name = "define_config_dataclass_basic",
    srcs = ["define_config_dataclass_basic.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//ml_collections/config_flags",
    ],
)

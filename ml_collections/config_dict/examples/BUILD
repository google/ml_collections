# Examples for ml_collections.config_dict

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_binary(
    name = "config_dict_advanced",
    srcs = ["config_dict_advanced.py"],
    data = ["config.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//ml_collections/config_flags",
        # pip: yaml
    ],
)

py_binary(
    name = "config_dict_basic",
    srcs = ["config_dict_basic.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//ml_collections",
    ],
)

py_binary(
    name = "frozen_config_dict",
    srcs = ["frozen_config_dict.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//ml_collections",
    ],
)

py_binary(
    name = "config_dict_initialization",
    srcs = ["config_dict_initialization.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//ml_collections",
    ],
)

py_binary(
    name = "config_dict_lock",
    srcs = ["config_dict_lock.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//ml_collections",
    ],
)

py_binary(
    name = "config_dict_placeholder",
    srcs = ["config_dict_placeholder.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//ml_collections",
    ],
)

py_binary(
    name = "field_reference",
    srcs = ["field_reference.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//ml_collections",
    ],
)

py_library(
    name = "examples_lib",
    srcs = [
        "config_dict_advanced.py",
        "config_dict_basic.py",
        "config_dict_initialization.py",
        "config_dict_lock.py",
        "config_dict_placeholder.py",
        "field_reference.py",
        "frozen_config_dict.py",
    ],
    data = ["config.py"],
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        "//ml_collections",
        "//ml_collections/config_dict",
        "//ml_collections/config_flags",
        # pip: yaml
    ],
)

py_test(
    name = "examples_test",
    srcs = ["examples_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":examples_lib",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//ml_collections",
    ],
)

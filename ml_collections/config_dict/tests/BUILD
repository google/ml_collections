# Package of tests for ml_collections.config_dict package.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_test(
    name = "config_dict_test",
    size = "small",
    srcs = [
        "config_dict_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//ml_collections",
        # pip: mock
        # pip: six
        # pip: yaml
    ],
)

py_test(
    name = "frozen_config_dict_test",
    size = "small",
    srcs = [
        "frozen_config_dict_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl/testing:absltest
        "//ml_collections",
    ],
)

py_test(
    name = "field_reference_test",
    size = "small",
    srcs = [
        "field_reference_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        "//ml_collections",
    ],
)

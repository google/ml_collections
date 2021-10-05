# Package of tests for ml_collections.library.config_flags package.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_test(
    name = "config_overriding_test",
    size = "small",
    srcs = [
        "config_overriding_test.py",
        "fieldreference_config.py",
        "mock_config.py",
    ],
    data = [
        "configdict_config.py",
        "ioerror_config.py",
        "mini_config.py",
        "parameterised_config.py",
        "typeerror_config.py",
        "valueerror_config.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl/flags
        # pip: absl/testing:absltest
        # pip: absl/testing:flagsaver
        # pip: absl/testing:parameterized
        "//ml_collections",
        "//ml_collections/config_flags",
        # pip: six
    ],
)

py_test(
    name = "dataclass_overriding_test",
    size = "small",
    srcs = ["dataclass_overriding_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl/flags
        # pip: absl/testing:absltest
        "//ml_collections/config_flags",
    ],
)

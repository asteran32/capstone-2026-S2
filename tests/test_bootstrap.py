"""Bootstrap checks for the installable package."""


def test_harness_package_imports() -> None:
    import harness

    assert harness.__name__ == "harness"

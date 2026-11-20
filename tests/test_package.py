from importlib.metadata import packages_distributions


def test_package_is_importable() -> None:
    import trading_research

    distributions = packages_distributions().get(trading_research.__name__, [])
    assert "event-driven-trading-research" in distributions

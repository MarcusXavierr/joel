import logging

import pytest

from joel.logs import configure_logging


@pytest.mark.usefixtures("restore_root_logger")
def test_sends_warnings_and_above_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging()
    logger = logging.getLogger("joel.test")

    logger.info("quiet")
    logger.warning("loud")

    err = capsys.readouterr().err
    assert "WARNING joel.test: loud" in err
    assert "quiet" not in err

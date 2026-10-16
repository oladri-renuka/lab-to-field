import pytest

from src.utils import RunLock


def test_second_live_process_is_refused(tmp_path):
    with RunLock(tmp_path):
        with pytest.raises(RuntimeError, match="already being trained"):
            with RunLock(tmp_path):
                pass
    assert not (tmp_path / ".lock").exists()


def test_stale_lock_is_ignored(tmp_path):
    (tmp_path / ".lock").write_text("999999999")  # no such process
    with RunLock(tmp_path):
        pass

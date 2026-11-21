from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event

import numpy as np
import pytest

from offline_face_recognition.core.config import Settings
from offline_face_recognition.gallery.session_store import (
    GallerySessionCapacityError,
    GallerySessionStore,
    GallerySessionStoreClosedError,
    GallerySessionUnavailableError,
    GalleryStoreInputError,
)


def _settings(**overrides):
    values = {
        "project_root": ".",
        "yunet_model_path": "yunet.onnx",
        "sface_model_path": "sface.onnx",
    }
    values.update(overrides)
    return Settings(**values)


def _identities(label="Asha"):
    return {label.casefold(): (label, np.eye(1, 128, 0, dtype=np.float32)[0])}


class FakeClock:
    def __init__(self):
        self.value = 500.0

    def monotonic(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def _store(settings=None, clock=None):
    active_clock = clock or FakeClock()
    store = GallerySessionStore(
        settings or _settings(),
        monotonic=active_clock.monotonic,
        utcnow=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return store, active_clock


def test_sessions_are_unpredictable_isolated_and_metadata_never_exposes_embeddings():
    store, _ = _store()
    first = store.create(_identities("Asha"))
    second = store.create(_identities("Ravi"))

    assert first.session_id != second.session_id
    assert len(first.session_id) >= 32
    assert first.identity_labels == ("Asha",)
    assert not hasattr(first, "embedding")
    with store._borrow(first.session_id) as entries:
        assert [entry.label for entry in entries] == ["Asha"]
        assert entries[0].embedding.flags.writeable is False
    with store._borrow(second.session_id) as entries:
        assert [entry.label for entry in entries] == ["Ravi"]
    with pytest.raises(GallerySessionUnavailableError):
        store._borrow("unknown-session").__enter__()
    store.close()


def test_session_expiration_uses_monotonic_time_and_zeroes_vectors():
    clock = FakeClock()
    store, _ = _store(_settings(gallery_session_ttl_seconds=10), clock)
    info = store.create(_identities())
    with store._borrow(info.session_id) as entries:
        stored_vector = entries[0].embedding
    assert np.linalg.norm(stored_vector) == pytest.approx(1.0)

    clock.advance(10.01)
    assert store.purge_expired() == 1
    assert store.active_session_count == 0
    assert np.count_nonzero(stored_vector) == 0
    with pytest.raises(GallerySessionUnavailableError):
        store.get_info(info.session_id)
    store.close()


def test_expiry_is_based_on_monotonic_clock_not_wall_clock():
    clock = FakeClock()
    store, _ = _store(_settings(gallery_session_ttl_seconds=10), clock)
    info = store.create(_identities())
    clock.advance(9.9)

    assert store.get_info(info.session_id).session_id == info.session_id
    clock.advance(0.2)
    with pytest.raises(GallerySessionUnavailableError):
        store.get_info(info.session_id)
    store.close()


def test_delete_waits_for_active_match_lease_then_clears_memory():
    store, _ = _store()
    info = store.create(_identities())
    entered = Event()
    release = Event()
    with store._borrow(info.session_id) as entries:
        stored_vector = entries[0].embedding

    def hold_lease():
        with store._borrow(info.session_id):
            entered.set()
            assert release.wait(timeout=3)

    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(hold_lease)
        assert entered.wait(timeout=2)
        deleter = pool.submit(store.delete, info.session_id)
        assert not deleter.done()
        release.set()
        reader.result(timeout=3)
        assert deleter.result(timeout=3)

    assert np.count_nonzero(stored_vector) == 0
    assert store.active_session_count == 0
    store.close()


def test_active_session_limit_and_close_clear_all_embeddings():
    store, _ = _store(_settings(max_active_gallery_sessions=2))
    infos = [store.create(_identities(f"Person{i}")) for i in range(2)]
    with pytest.raises(GallerySessionCapacityError):
        store.create(_identities("Third"))
    references = []
    for info in infos:
        with store._borrow(info.session_id) as entries:
            references.append(entries[0].embedding)

    store.close()

    assert all(np.count_nonzero(vector) == 0 for vector in references)
    assert store.active_session_count == 0
    with pytest.raises(GallerySessionStoreClosedError):
        store.create(_identities("AfterClose"))


@pytest.mark.parametrize(
    "identities",
    [
        {},
        {"person": ("Person", np.zeros(128, np.float32))},
        {"person": ("Person", np.ones(127, np.float32))},
    ],
)
def test_invalid_gallery_entries_are_rejected(identities):
    store, _ = _store()
    with pytest.raises(GalleryStoreInputError):
        store.create(identities)
    assert store.active_session_count == 0
    store.close()

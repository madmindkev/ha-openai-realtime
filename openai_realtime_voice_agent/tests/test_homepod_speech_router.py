"""Deterministic, network-free unit tests for HomePodSpeechRouter's HA transport.

These tests import the real ``app.homepod_speech_router.HomePodSpeechRouter``
and simulate Home Assistant by patching ``urllib.request.urlopen`` at the
module level. No real HTTP request, Home Assistant instance, OpenAI call, or
audio device is ever touched.
"""

import json
import unittest
import urllib.error
from unittest.mock import patch

from app.homepod_speech_router import HomePodSpeechRouter

TARGET_ENTITY = "media_player.salon_test_homepod"
TTS_ENTITY = "tts.test_morgan"
HA_API_BASE = "http://supervisor/core/api"
FAKE_TOKEN = "unit-test-token"


class _FakeHTTPResponse:
    """Minimal stand-in for the object returned by ``urlopen(...)``."""

    def __init__(self, status: int = 200, body: bytes = b""):
        self._status = status
        self._body = body

    def getcode(self):
        return self._status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _build_router(**overrides) -> HomePodSpeechRouter:
    kwargs = {
        "target_entity": TARGET_ENTITY,
        "tts_entity": TTS_ENTITY,
        "ha_api_base": HA_API_BASE,
        "timeout_seconds": 1.0,
    }
    kwargs.update(overrides)
    return HomePodSpeechRouter(**kwargs)


def _request_json_body(call) -> dict:
    """Extract the JSON body sent on a mocked urlopen(request, ...) call."""
    request = call.args[0]
    return json.loads(request.data.decode("utf-8"))


def _request_url(call) -> str:
    request = call.args[0]
    return request.full_url


class HomePodSpeechRouterTransportTests(unittest.IsolatedAsyncioTestCase):
    """Verify the exact Home Assistant payloads for both playback paths."""

    def setUp(self):
        # Force a known, non-empty token regardless of the host environment,
        # and stub time.sleep so retry back-off never slows the suite down.
        self._env_patcher = patch.dict(
            "os.environ",
            {"LONGLIVED_TOKEN": FAKE_TOKEN, "SUPERVISOR_TOKEN": ""},
        )
        self._env_patcher.start()
        self.addCleanup(self._env_patcher.stop)

        self._sleep_patcher = patch("app.homepod_speech_router.time.sleep")
        self._sleep_patcher.start()
        self.addCleanup(self._sleep_patcher.stop)

    async def test_fast_path_play_media_targets_constructor_entity(self):
        """The fast path must send play_media to exactly target_entity."""
        router = _build_router()

        stream_url = "http://example.invalid/morgan-stream.mp3"
        responses = [
            _FakeHTTPResponse(200, json.dumps({"url": stream_url}).encode("utf-8")),
            _FakeHTTPResponse(200, b""),
        ]

        with patch(
            "app.homepod_speech_router.urllib.request.urlopen",
            side_effect=responses,
        ) as mock_urlopen:
            routed, _elapsed = await router._speak_on_homepod("Bonjour depuis le salon")

        self.assertTrue(routed)
        self.assertEqual(mock_urlopen.call_count, 2)

        get_url_call, play_media_call = mock_urlopen.call_args_list

        self.assertTrue(_request_url(get_url_call).endswith("/tts_get_url"))
        get_url_body = _request_json_body(get_url_call)
        self.assertEqual(get_url_body["engine_id"], TTS_ENTITY)

        self.assertTrue(
            _request_url(play_media_call).endswith(
                "/services/media_player/play_media"
            )
        )
        play_media_body = _request_json_body(play_media_call)
        self.assertEqual(play_media_body["entity_id"], TARGET_ENTITY)
        self.assertEqual(play_media_body["media_content_id"], stream_url)
        self.assertEqual(play_media_body["media_content_type"], "music")
        self.assertTrue(play_media_body["announce"])

    async def test_fallback_tts_speak_uses_target_entity_as_media_player(self):
        """When the fast path fails, tts.speak must reuse target_entity as
        media_player_entity_id — not some other/default entity."""
        router = _build_router()

        # tts_get_url exhausts its 3 attempts with a network error, forcing
        # the router onto the legacy tts.speak fallback, which then succeeds.
        responses = [
            urllib.error.URLError("simulated network failure"),
            urllib.error.URLError("simulated network failure"),
            urllib.error.URLError("simulated network failure"),
            _FakeHTTPResponse(200, b""),
        ]

        with patch(
            "app.homepod_speech_router.urllib.request.urlopen",
            side_effect=responses,
        ) as mock_urlopen:
            routed, _elapsed = await router._speak_on_homepod("Bonjour depuis le salon")

        self.assertTrue(routed)
        self.assertEqual(mock_urlopen.call_count, 4)

        tts_speak_call = mock_urlopen.call_args_list[-1]
        self.assertTrue(_request_url(tts_speak_call).endswith("/services/tts/speak"))

        tts_speak_body = _request_json_body(tts_speak_call)
        self.assertEqual(tts_speak_body["entity_id"], TTS_ENTITY)
        self.assertEqual(tts_speak_body["media_player_entity_id"], TARGET_ENTITY)
        self.assertEqual(tts_speak_body["message"], "Bonjour depuis le salon")

    async def test_both_paths_failing_reports_not_routed(self):
        """If Home Assistant rejects both transports, _speak_on_homepod must
        report failure so the caller falls back to local Voice PE audio."""
        router = _build_router()

        responses = [urllib.error.URLError("simulated network failure")] * 6

        with patch(
            "app.homepod_speech_router.urllib.request.urlopen",
            side_effect=responses,
        ):
            routed, _elapsed = await router._speak_on_homepod("Bonjour")

        self.assertFalse(routed)

    def test_constructor_defaults_are_not_silently_used_when_overridden(self):
        """Sanity check: passing target_entity actually overrides the default
        so the transport tests above are exercising the intended value, not
        HomePodSpeechRouter's built-in default entity."""
        default_router = HomePodSpeechRouter()
        router = _build_router()
        self.assertEqual(router._target_entity, TARGET_ENTITY)
        self.assertNotEqual(router._target_entity, default_router._target_entity)


if __name__ == "__main__":
    unittest.main()

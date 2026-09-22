#!/usr/bin/env python3
"""Tests the job executor main.py runs its Room jobs with.

Cloud providers always get JobExecutorType.PROCESS. For the local providers
(vosk, sherpa) the `job_executor` property of agent-speech-processing.yaml
decides between `thread` (default: every Room a thread of one process sharing
one model copy) and `process` (every Room in its own process with its own model
copy); the JOB_EXECUTOR_TYPE env var overrides for testing. Under `process`,
prewarm() loads the local model in every job subprocess so the first Room of a
warm process does not pay for it. Nothing here loads a model or a VAD.
"""

import unittest
from unittest import mock

from livekit.agents import JobExecutorType

import main

THREAD, PROCESS = JobExecutorType.THREAD, JobExecutorType.PROCESS


class ResolveJobExecutorTypeTests(unittest.TestCase):
    def test_local_providers_default_to_one_process_for_all_rooms(self):
        for provider in ("vosk", "sherpa"):
            self.assertEqual(main._resolve_job_executor_type(provider, "thread"), THREAD)
            # an empty `job_executor:` line in the YAML reaches us as None
            self.assertEqual(main._resolve_job_executor_type(provider, None), THREAD)

    def test_local_providers_can_run_one_process_per_room(self):
        for provider in ("vosk", "sherpa"):
            self.assertEqual(main._resolve_job_executor_type(provider, "process"), PROCESS)
        self.assertEqual(main._resolve_job_executor_type("vosk", " Process\n"), PROCESS)

    def test_cloud_providers_always_get_one_process_per_room(self):
        for configured in ("thread", "process", None):
            self.assertEqual(main._resolve_job_executor_type("aws", configured), PROCESS)
            self.assertEqual(main._resolve_job_executor_type(None, configured), PROCESS)

    def test_an_invalid_property_value_is_rejected(self):
        for bad in ("auto", "", "yes", "threads", 3):
            with self.assertRaises(ValueError) as error:
                main._resolve_job_executor_type("vosk", bad)
            self.assertIn("job_executor must be 'thread' or 'process'", str(error.exception))
            self.assertIn(str(bad), str(error.exception))
        # rejected before any override is looked at
        with self.assertRaises(ValueError):
            main._resolve_job_executor_type("aws", "auto", "thread")

    def test_env_override_wins_for_any_provider(self):
        with self.assertLogs(level="INFO") as logs:
            self.assertEqual(main._resolve_job_executor_type("aws", "process", "thread"), THREAD)
            self.assertEqual(main._resolve_job_executor_type("vosk", "thread", "process"), PROCESS)
            self.assertEqual(main._resolve_job_executor_type("sherpa", "process", " THREAD "), THREAD)
        self.assertEqual(len(logs.output), 3)
        self.assertTrue(all("JOB_EXECUTOR_TYPE" in line for line in logs.output))

    def test_unknown_env_override_is_ignored_with_a_warning(self):
        with self.assertLogs(level="WARNING") as logs:
            self.assertEqual(main._resolve_job_executor_type("vosk", "thread", "banana"), THREAD)
            self.assertEqual(main._resolve_job_executor_type("vosk", "process", "banana"), PROCESS)
        self.assertEqual(len(logs.output), 2)
        self.assertIn("banana", logs.output[0])

    def test_local_providers_are_exactly_vosk_and_sherpa(self):
        self.assertEqual(main.LOCAL_STT_PROVIDERS, ("vosk", "sherpa"))


class PrewarmModelPreloadTests(unittest.TestCase):
    """prewarm() in a job subprocess vs the worker process."""

    CONFIG = {"live_captions": {"provider": "sherpa", "sherpa": {"model": "m"}}}

    def run_prewarm(self, *, in_job_subprocess, config_error=None):
        proc = mock.Mock()
        proc.userdata = {}
        get_instance = mock.Mock()
        if config_error is None:
            get_instance.return_value.get_agent_config.return_value = self.CONFIG
        else:
            get_instance.return_value.get_agent_config.side_effect = config_error
        vad = object()
        with mock.patch.object(main, "_is_job_subprocess", return_value=in_job_subprocess), mock.patch.object(
            main, "_early_log_handler", None
        ), mock.patch.object(main, "_install_ffi_panic_guard") as panic_guard, mock.patch(
            "stt_impl._get_cached_silero_vad", return_value=vad
        ), mock.patch.object(main, "_preload_vosk_model") as vosk, mock.patch.object(
            main, "_preload_sherpa_model"
        ) as sherpa, mock.patch.object(main.OpenViduAgent, "get_instance", get_instance):
            main.prewarm(proc)
        self.assertIs(proc.userdata["vad"], vad)
        return get_instance, vosk, sherpa, panic_guard

    def test_a_job_subprocess_preloads_the_local_model(self):
        get_instance, vosk, sherpa, panic_guard = self.run_prewarm(in_job_subprocess=True)
        get_instance.assert_called()
        vosk.assert_called_once_with(self.CONFIG)
        sherpa.assert_called_once_with(self.CONFIG)
        panic_guard.assert_called_once_with()

    def test_the_worker_process_leaves_the_preload_to_main(self):
        get_instance, vosk, sherpa, panic_guard = self.run_prewarm(in_job_subprocess=False)
        get_instance.assert_not_called()
        vosk.assert_not_called()
        sherpa.assert_not_called()
        panic_guard.assert_not_called()

    def test_an_unavailable_configuration_is_only_a_warning(self):
        # the config loader calls exit() on a broken file: prewarm must survive it
        with self.assertLogs(level="WARNING") as logs:
            _, vosk, sherpa, _ = self.run_prewarm(in_job_subprocess=True, config_error=SystemExit(1))
        vosk.assert_not_called()
        sherpa.assert_not_called()
        self.assertTrue(any("the ASR model will load with the first Room" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()

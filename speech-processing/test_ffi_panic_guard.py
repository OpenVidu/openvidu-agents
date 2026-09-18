#!/usr/bin/env python3
"""Guards `_install_ffi_panic_guard()` in main.py against upstream drift.

livekit.rtc answers *every* FFI panic by terminating the process:

    elif which == "panic":
        print("FFI Panic: ", event.panic.message, file=sys.stderr, flush=True)
        # We are in a unrecoverable state, terminate the process
        os.kill(os.getpid(), signal.SIGTERM)

Under JobExecutorType.THREAD every room shares one process, so a panic raised by
a single job takes the whole agent container down with it. One such panic is
reachable from ordinary operation -- a room deleted while its job is still
connecting leaves the Rust side waiting for a ReadyForRoomEventRequest that never
comes -- so main.py intercepts that specific message and lets the process live.

These tests replay the two statements of that branch verbatim against the
installed SDK. They fail if upstream renames the panic message, restructures the
branch, or stops routing it through the module globals the guard patches.
"""

import importlib.util
import os
import signal
import sys
import unittest

_MAIN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")

RECOVERABLE_MESSAGE = (
    "invalid request: timed out waiting for ReadyForRoomEventRequest "
    "after ConnectCallback (room_handle=7)"
)
UNKNOWN_MESSAGE = "unrecoverable: ffi runtime corrupted"


def _load_main():
    """Import main.py under its own name, without running the __main__ block."""
    spec = importlib.util.spec_from_file_location("agentmain_for_tests", _MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PanicBranchHarness:
    """Installs the guard over a probe and replays the panic branch."""

    def __init__(self):
        self.delegated: list[int] = []
        self.main = _load_main()

        import livekit.rtc._ffi_client as ffi_client

        self.ffi_client = ffi_client
        # The guard captures _ffi_client.os.kill when it installs, so the probe
        # has to be in place first: whatever the guard decides to let through
        # lands here instead of actually signalling the test runner.
        self._real_kill = ffi_client.os.kill
        ffi_client.os.kill = lambda pid, sig: self.delegated.append(sig)
        self.main._install_ffi_panic_guard()

    def panic(self, message):
        """Byte-for-byte the two statements of ffi_event_callback's panic branch."""
        print("FFI Panic: ", message, file=self.ffi_client.sys.stderr, flush=True)
        self.ffi_client.os.kill(self.ffi_client.os.getpid(), signal.SIGTERM)

    def restore(self):
        self.ffi_client.os = os
        self.ffi_client.sys = sys


class TestFfiPanicGuard(unittest.TestCase):
    def setUp(self):
        self.h = _PanicBranchHarness()
        self.addCleanup(self.h.restore)

    def test_recoverable_panic_does_not_terminate_the_process(self):
        """The room is already gone; every other job must keep running."""
        self.h.panic(RECOVERABLE_MESSAGE)
        self.assertEqual(
            self.h.delegated,
            [],
            "the SIGTERM from a recoverable FFI panic reached os.kill: a single "
            "room's panic would terminate the whole agent container. Check that "
            "main._RECOVERABLE_FFI_PANIC_MARKERS still matches the message the "
            "installed livekit.rtc prints.",
        )

    def test_unknown_panic_still_terminates_the_process(self):
        """Fail-fast stays the default for panics we have not reasoned about."""
        self.h.panic(UNKNOWN_MESSAGE)
        self.assertEqual(
            self.h.delegated,
            [signal.SIGTERM],
            "an unrecognized FFI panic was swallowed; the guard must only "
            "recover from the messages listed in _RECOVERABLE_FFI_PANIC_MARKERS.",
        )

    def test_repeated_recoverable_panics_are_each_swallowed(self):
        """One suppression per panic: the counter must not run out or leak."""
        for _ in range(3):
            self.h.panic(RECOVERABLE_MESSAGE)
        self.assertEqual(self.h.delegated, [])

    def test_real_sigterm_still_shuts_the_agent_down(self):
        """A SIGTERM with no panic behind it is the orchestrator asking to stop."""
        self.h.ffi_client.os.kill(os.getpid(), signal.SIGTERM)
        self.assertEqual(
            self.h.delegated,
            [signal.SIGTERM],
            "the guard swallowed a genuine shutdown signal, so the container "
            "would ignore `docker stop` / pod termination.",
        )

    def test_recoverable_panic_does_not_mask_a_later_real_sigterm(self):
        """Suppression is consumed by its own panic, not by the next signal."""
        self.h.panic(RECOVERABLE_MESSAGE)
        self.h.ffi_client.os.kill(os.getpid(), signal.SIGTERM)
        self.assertEqual(
            self.h.delegated,
            [signal.SIGTERM],
            "a recoverable panic left suppression armed, so the next real "
            "SIGTERM was swallowed too.",
        )


class TestPanicBranchStillLooksLikeWeThink(unittest.TestCase):
    """Tripwires for the upstream code the guard is shaped around."""

    @staticmethod
    def _ffi_client_source() -> str:
        """The module's text.

        `ffi_event_callback` is wrapped in @ctypes.CFUNCTYPE, so it is a
        CFunctionType at runtime and inspect.getsource() cannot read it; read
        the file the module was loaded from instead.
        """
        import livekit.rtc._ffi_client as ffi_client

        with open(ffi_client.__file__, encoding="utf-8") as f:
            return f.read()

    def test_ffi_client_still_kills_the_process_on_panic(self):
        """If upstream stops killing, the whole guard can be deleted."""
        source = self._ffi_client_source()
        self.assertIn(
            'which == "panic"',
            source,
            "livekit.rtc.ffi_event_callback no longer has a panic branch; "
            "re-read it and check whether main._install_ffi_panic_guard is "
            "still needed.",
        )
        self.assertIn(
            "os.kill(os.getpid(), signal.SIGTERM)",
            source,
            "the panic branch no longer terminates the process with SIGTERM. "
            "The guard patches that exact call, so it is now a no-op: re-read "
            "the branch and update or remove it.",
        )

    def test_panic_message_is_printed_through_module_sys(self):
        """The guard reads the message via _ffi_client's `sys`, not global stderr."""
        source = self._ffi_client_source()
        self.assertIn(
            "file=sys.stderr",
            source,
            "the panic message is no longer printed to sys.stderr, so the guard "
            "cannot classify the panic and would suppress nothing.",
        )

    def test_kill_is_reached_through_the_module_global_os(self):
        """The guard patches `_ffi_client.os`, so the kill must resolve through it.

        A switch to `signal.raise_signal(...)`, `os._exit(...)` or a directly
        imported `kill` would bypass the proxy entirely: the guard would suppress
        nothing and the container would start dying again with every test green.
        """
        source = self._ffi_client_source()
        self.assertNotIn(
            "signal.raise_signal",
            source,
            "livekit.rtc now raises the signal directly instead of calling "
            "os.kill, which bypasses the guard's `os` proxy.",
        )
        self.assertNotIn(
            "os._exit",
            source,
            "livekit.rtc now calls os._exit, which cannot be intercepted by a "
            "signal handler or by the guard. Re-read the panic branch.",
        )


class TestRecoverablePanicMessageStillMatches(unittest.TestCase):
    """The message match is the guard's weakest link: nothing else pins it.

    _RECOVERABLE_FFI_PANIC_MARKERS is matched against text produced by the Rust
    side, which no Python signature covers. If that wording is reworded upstream
    the guard silently stops suppressing -- the failure mode is a container that
    dies again in production while every other test here still passes. These
    tests tie the marker to the SDK artifacts that carry the wording.
    """

    def setUp(self):
        self.main = _load_main()

    def test_markers_are_non_empty_and_specific(self):
        markers = self.main._RECOVERABLE_FFI_PANIC_MARKERS
        self.assertTrue(markers, "no recoverable panic markers configured")
        for marker in markers:
            with self.subTest(marker=marker):
                # A marker short enough to match unrelated panics would make the
                # guard swallow failures it has not reasoned about.
                self.assertGreaterEqual(
                    len(marker),
                    20,
                    f"marker {marker!r} is too generic; it risks suppressing "
                    f"panics that really are unrecoverable.",
                )

    def test_marker_text_is_present_in_the_shipped_ffi_binary(self):
        """The Rust binary embeds its panic strings; grep for ours.

        This is what actually catches a rewording: the marker disappears from
        the shipped library before it disappears from production logs.
        """
        import livekit.rtc as rtc_pkg

        resources = os.path.join(os.path.dirname(rtc_pkg.__file__), "resources")
        if not os.path.isdir(resources):
            self.skipTest(f"no livekit.rtc resources directory at {resources}")

        libraries = [
            os.path.join(resources, name)
            for name in os.listdir(resources)
            if name.endswith((".so", ".dylib", ".dll"))
        ]
        if not libraries:
            self.skipTest(f"no FFI shared library found in {resources}")

        for marker in self.main._RECOVERABLE_FFI_PANIC_MARKERS:
            needle = marker.encode()
            found_in = [
                lib
                for lib in libraries
                if needle in open(lib, "rb").read()  # noqa: SIM115
            ]
            with self.subTest(marker=marker):
                self.assertTrue(
                    found_in,
                    f"the panic text {marker!r} no longer appears in the "
                    f"shipped livekit FFI library "
                    f"({', '.join(os.path.basename(x) for x in libraries)}). "
                    f"Upstream reworded or removed it, so "
                    f"main._RECOVERABLE_FFI_PANIC_MARKERS no longer matches and "
                    f"the guard suppresses nothing: a single room's panic will "
                    f"again terminate the whole agent container. Find the new "
                    f"wording and update the marker.",
                )


class TestGuardIsActuallyInstalled(unittest.TestCase):
    """The guard only helps if main.py still calls it on both executor paths.

    Every other test installs it by hand, so deleting a call site in main.py
    would leave the suite green while production regains the original bug.
    """

    @staticmethod
    def _main_source() -> str:
        with open(_MAIN_PY, encoding="utf-8") as f:
            return f.read()

    def test_installed_for_the_thread_executor(self):
        """__main__ path: one process hosts every room, so this one matters most."""
        import ast

        tree = ast.parse(self._main_source())
        main_blocks = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "__name__ == '__main__'"
        ]
        self.assertTrue(main_blocks, "main.py has no `if __name__ == '__main__'` block")
        called = any(
            isinstance(inner, ast.Call)
            and getattr(inner.func, "id", None) == "_install_ffi_panic_guard"
            for block in main_blocks
            for inner in ast.walk(block)
        )
        self.assertTrue(
            called,
            "main.py's __main__ block no longer calls _install_ffi_panic_guard(): "
            "under JobExecutorType.THREAD one room's FFI panic will terminate the "
            "whole agent container again.",
        )

    def test_installed_for_job_subprocesses(self):
        """prewarm() runs in each PROCESS-executor child, which imports its own rtc."""
        import ast

        tree = ast.parse(self._main_source())
        prewarm = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "prewarm"
            ),
            None,
        )
        self.assertIsNotNone(prewarm, "main.py no longer defines prewarm()")
        called = any(
            isinstance(inner, ast.Call)
            and getattr(inner.func, "id", None) == "_install_ffi_panic_guard"
            for inner in ast.walk(prewarm)
        )
        self.assertTrue(
            called,
            "prewarm() no longer calls _install_ffi_panic_guard(), so job "
            "subprocesses run unguarded.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

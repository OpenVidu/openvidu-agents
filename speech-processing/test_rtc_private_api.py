#!/usr/bin/env python3
"""Guards the private livekit.rtc API that main.py's FFI leak workaround depends on.

`_release_room_ffi_subscription` reaches into `rtc.Room` internals to undo a leak
the SDK leaves behind on a server-side room close (see its docstring). Every one of
those reads is defensive -- `getattr(..., None)`, `contextlib.suppress(Exception)` --
so when a bump renames one of them the workaround does not fail, it silently stops
working. That already happened once: rtc 1.1.13 -> 1.1.18 (pulled in by livekit-agents
1.6.5 -> 1.8.1) replaced the async `_drain_data_stream_tasks()` with the two sync
`_error_stream_readers()` / `_dispose_open_stream_writers()`, and the old name simply
resolved to None from then on.

These tests turn that silence into a `pip install`-time failure. They assert against
the *installed* livekit rtc, so they need no deployment, no room and no agent job.
"""

import ast
import inspect
import textwrap
import unittest

from livekit import rtc
from livekit.rtc._ffi_client import FfiClient

import main


def _listen_task_cleanup_calls() -> list[str]:
    """The `self.<name>()` calls at the tail of `rtc.Room._listen_task`.

    That tail is the cleanup the task runs after breaking out of its event loop on
    'eos' -- exactly what `_release_room_ffi_subscription` replays by hand when it
    cancels a listen task that will never see one.
    """
    source = textwrap.dedent(inspect.getsource(rtc.Room._listen_task))
    func = ast.parse(source).body[0]

    loops = [i for i, node in enumerate(func.body) if isinstance(node, ast.While)]
    if not loops:
        raise AssertionError(
            "rtc.Room._listen_task no longer has a top-level event loop, so its "
            "cleanup tail cannot be located. Re-read the method and review "
            "main.py's _release_room_ffi_subscription against it."
        )

    names: list[str] = []
    for statement in func.body[loops[-1] + 1 :]:
        if not isinstance(statement, ast.Expr):
            continue
        call = statement.value
        if isinstance(call, ast.Await):
            call = call.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
        ):
            names.append(call.func.attr)
    return names


class TestRoomListenTaskCleanups(unittest.TestCase):
    """main._ROOM_LISTEN_TASK_CLEANUPS vs. the installed rtc.Room."""

    def test_every_cleanup_name_exists_on_room(self):
        """A name that is gone is skipped silently by the workaround."""
        self.assertGreater(len(main._ROOM_LISTEN_TASK_CLEANUPS), 0)
        for name in main._ROOM_LISTEN_TASK_CLEANUPS:
            with self.subTest(cleanup=name):
                self.assertTrue(
                    hasattr(rtc.Room, name),
                    f"rtc.Room has no '{name}': main._ROOM_LISTEN_TASK_CLEANUPS is "
                    f"stale, so _release_room_ffi_subscription skips this cleanup "
                    f"without a word. Check the tail of rtc.Room._listen_task "
                    f"(rtc {_rtc_version()}) for what replaced it.",
                )
                self.assertTrue(
                    callable(getattr(rtc.Room, name)),
                    f"rtc.Room.{name} is no longer callable",
                )

    def test_cleanup_names_match_the_listen_task_tail(self):
        """The workaround must replay what the listen task itself would have run.

        Deliberately strict: a cleanup added upstream is one the workaround would
        stop performing, which is the same silent failure as a rename.
        """
        upstream = _listen_task_cleanup_calls()
        self.assertEqual(
            list(main._ROOM_LISTEN_TASK_CLEANUPS),
            upstream,
            f"main._ROOM_LISTEN_TASK_CLEANUPS is out of sync with the tail of "
            f"rtc.Room._listen_task in rtc {_rtc_version()}. Upstream now runs "
            f"{upstream}; update the tuple (order included) so a remotely closed "
            f"room gets the same cleanup a normal 'eos' close would.",
        )

    def test_cleanups_may_be_sync_or_async(self):
        """Both shapes must stay handled: 1.1.13's was async, 1.1.18's are sync."""
        shapes = {
            name: inspect.iscoroutinefunction(getattr(rtc.Room, name))
            for name in main._ROOM_LISTEN_TASK_CLEANUPS
            if hasattr(rtc.Room, name)
        }
        self.assertTrue(shapes, "no cleanup resolved on rtc.Room")
        # The loop calls first and awaits only an awaitable result, so any mix is
        # fine; this records which is which to make a future flip visible.
        print(f"\ncleanup shapes (True = coroutine function): {shapes}")


class TestRoomAttributesUsedByTheWorkaround(unittest.TestCase):
    """The rest of the private surface _release_room_ffi_subscription touches."""

    def test_room_exposes_the_public_entry_points(self):
        for name in ("isconnected", "disconnect", "name"):
            with self.subTest(member=name):
                self.assertTrue(hasattr(rtc.Room, name), f"rtc.Room has no '{name}'")

    def test_room_assigns_the_private_attributes(self):
        """`_task` and `_ffi_queue` only exist after connect(), so check the source.

        Both are read with `getattr(room, ..., None)`: a rename turns the whole
        cancel-and-unsubscribe block into a no-op.
        """
        source = inspect.getsource(rtc.Room)
        for name in ("_task", "_ffi_queue"):
            with self.subTest(attribute=name):
                self.assertRegex(
                    source,
                    rf"self\.{name}\s*[:=]",
                    f"rtc.Room never assigns self.{name} in rtc {_rtc_version()}; "
                    f"_release_room_ffi_subscription reads it with a getattr default "
                    f"and would silently do nothing.",
                )


class TestFfiQueueInternals(unittest.TestCase):
    """The workaround inspects FfiClient.instance.queue to find a leaked subscription."""

    def test_queue_exposes_the_members_the_workaround_uses(self):
        queue = FfiClient.instance.queue
        for name in ("_lock", "_subscribers", "unsubscribe"):
            with self.subTest(member=name):
                self.assertTrue(
                    hasattr(queue, name),
                    f"{type(queue).__name__} has no '{name}': the unsubscribe branch "
                    f"of _release_room_ffi_subscription would raise into its "
                    f"`except Exception` and only log a warning.",
                )

    def test_subscriber_entries_unpack_into_three(self):
        """`any(q is ffi_queue for q, _, _ in queue._subscribers)` pins the arity.

        A different tuple width raises ValueError inside the workaround's broad
        `except`, which downgrades the leak fix to a log line.
        """
        queue = FfiClient.instance.queue
        subscription = queue.subscribe()
        try:
            with queue._lock:
                entries = list(queue._subscribers)
            match = [entry for entry in entries if entry[0] is subscription]
            self.assertEqual(
                len(match),
                1,
                "a fresh subscription is not findable as the first element of a "
                "_subscribers entry",
            )
            self.assertEqual(
                len(match[0]),
                3,
                f"_subscribers entries are {len(match[0])}-tuples in rtc "
                f"{_rtc_version()}; _release_room_ffi_subscription unpacks them as "
                f"`for q, _, _ in ...`.",
            )
        finally:
            queue.unsubscribe(subscription)


class TestWorkaroundIsStillNeeded(unittest.TestCase):
    """Tripwire for the day upstream fixes this and the workaround can be deleted."""

    def test_disconnect_still_returns_early_when_not_connected(self):
        """`Room.disconnect()` bailing out before its own unsubscribe is the leak.

        A failure here is good news: re-read `_release_room_ffi_subscription`'s
        docstring and check whether the whole function can go.
        """
        source = textwrap.dedent(inspect.getsource(rtc.Room.disconnect))
        func = ast.parse(source).body[0]
        first = func.body[0] if not _is_docstring(func.body[0]) else func.body[1]

        self.assertIsInstance(
            first,
            ast.If,
            "rtc.Room.disconnect no longer starts with a guard clause; it may now "
            "clean up on a remote close, which would make "
            "main._release_room_ffi_subscription unnecessary.",
        )
        self.assertTrue(
            any(isinstance(node, ast.Return) for node in ast.walk(first)),
            "rtc.Room.disconnect's opening guard no longer returns early; check "
            "whether it now unsubscribes the FFI queue on a remote close.",
        )
        self.assertRegex(
            ast.unparse(first.test),
            r"isconnected",
            "rtc.Room.disconnect's early return is no longer keyed on "
            "isconnected(); re-read it against the workaround's assumptions.",
        )


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)


def _rtc_version() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("livekit")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


if __name__ == "__main__":
    unittest.main(verbosity=2)

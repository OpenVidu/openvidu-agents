#!/usr/bin/env python3
"""Tests that every job is accepted with the participant attribute that marks an OpenVidu agent."""

import asyncio
import unittest
from unittest import mock

import main


class AcceptJobTests(unittest.TestCase):
    def test_every_job_is_accepted_as_an_openvidu_agent(self):
        req = mock.AsyncMock()

        # A private loop: IsolatedAsyncioTestCase and asyncio.run() leave the main thread without a
        # current loop, which breaks later tests that call asyncio.get_event_loop().
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(main.accept_job(req))
        finally:
            loop.close()

        # only attributes: identity, name and metadata keep the defaults of livekit-agents
        req.accept.assert_awaited_once_with(attributes={"openvidu.agent": "speech-processing"})


if __name__ == "__main__":
    unittest.main()

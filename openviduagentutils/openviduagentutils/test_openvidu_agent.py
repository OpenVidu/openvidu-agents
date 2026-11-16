import unittest

from .openvidu_agent import OPENVIDU_AGENT_ATTRIBUTE, openvidu_agent_attributes


class TestOpenViduAgentAttributes(unittest.TestCase):
    def test_the_key_is_the_one_the_server_reads(self):
        # openvidu-livekit's analytics look for this exact key: changing it hides every agent's mark
        self.assertEqual(OPENVIDU_AGENT_ATTRIBUTE, "openvidu.agent")

    def test_the_attributes_carry_only_the_agent_type(self):
        self.assertEqual(
            openvidu_agent_attributes("speech-processing"),
            {"openvidu.agent": "speech-processing"},
        )


if __name__ == "__main__":
    unittest.main()

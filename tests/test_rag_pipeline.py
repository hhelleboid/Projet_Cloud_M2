import unittest
import os
from app.query import Config

class TestRAGFunctions(unittest.TestCase):
    """
    Test suite to check critical configuration and code integrity 
    before building and deploying Docker images.
    """

    def test_a_ollama_host_is_set_for_azure(self):
        """
        Verifies that the Ollama host is configured to use the Azure-compliant 
        internal hostname (ollama-server) by default, which is necessary for ACA.
        This confirms the code is ready for cloud deployment.
        """
        expected_host_suffix = ":11434"
        
        self.assertIn("ollama-server", Config.LLM_BASE_URL, 
                      "Error: LLM_BASE_URL must be set to 'http://ollama-server:11434' for ACA deployment.")
        self.assertTrue(Config.LLM_BASE_URL.endswith(expected_host_suffix),
                        f"Error: LLM_BASE_URL must end with '{expected_host_suffix}'.")


    def test_b_environment_variables_exist(self):
        """
        Verifies that the CI runner has access to the ACR Name needed for pushing images.
        """
        self.assertTrue(os.getenv("ACR_NAME") is not None, "ACR_NAME environment variable must be set in the CI workflow.")
        

if __name__ == '__main__':
    unittest.main()

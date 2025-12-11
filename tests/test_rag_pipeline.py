import unittest
import os
from app.query import Config
import pytest

class TestRAGFunctions(unittest.TestCase):
    
    def test_a_ollama_host_is_set_for_azure(self):
        """
        Vérifie que l'URL du backend est configurée correctement.
        """
        # On récupère l'URL configurée
        base_url = Config.LLM_BASE_URL
        
        # Vérifier le protocole
        self.assertTrue(base_url.startswith("http://"), "L'URL doit commencer par http://")
        
        # Vérifier le nom d'hôte 
        valid_hosts = ["ca-backend", "localhost"]
        
        is_valid_host = any(host in base_url for host in valid_hosts)
        
        self.assertTrue(is_valid_host, 
                      f"L'URL '{base_url}' ne contient aucun nom d'hôte valide (attendu: ca-backend pour Azure).")
    
    def test_import_streamlit(self):
        """Vérifie que les dépendances sont bien installées"""
        try:
            import streamlit
            import langchain
            assert True
        except ImportError:
            pytest.fail("Une librairie manque à l'appel")

if __name__ == '__main__':
    unittest.main()


import logging
import unittest
from typing import Dict, List, Any
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('test_results.log')]
)
logger = logging.getLogger(__name__)

# Import the functions being tested
try:
    from agents.podcast_agent import execute_rag_response
    import asyncio
    from agents.entry import chatbot_entry
    from langchain_core.messages import AIMessage, HumanMessage
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)


class TestData:
    """Test data for podcast agent tests."""
    
    # RAG test data
    RAG_QUERIES = [
        {
            "query": "Is Israel doing the correct thing attacking Gaza?",
            "user_intent": "question",
            "output_emotion": "neutral",
            "expected_result_type": str,
            "description": "Test controversial geopolitical question"
        },
        {
            "query": "What are the benefits of machine learning?",
            "user_intent": "information",
            "output_emotion": "positive",
            "expected_result_type": str,
            "description": "Test informational query about technology"
        },
        {
            "query": "",  # Empty query to test error handling
            "user_intent": "question",
            "output_emotion": "neutral",
            "expected_result_type": str,
            "description": "Test empty query handling"
        }
    ]
    
    # Chatbot test data
    CHATBOT_TEST_CASES = [
        {
            "query": "What is the podcast about?",
            "history": [
                HumanMessage(content="What is the podcast about?"),
                AIMessage(content="The podcast is about AI and machine learning"),
                HumanMessage(content="How has machine learning evolved over the years?"),
                AIMessage(content="Machine learning has evolved significantly over the years, with advancements in deep learning, reinforcement learning, and natural language processing."),
                HumanMessage(content="What are the key applications of AI in healthcare?"),
                AIMessage(content="AI is used in healthcare for medical imaging, drug discovery, personalized treatment, and predictive analytics."),
            ],
            "facts": "The podcast is about AI and machine learning",
            "description": "Test query about podcast topic with history"
        },
        {
            "query": "What was the last topic discussed?",
            "history": [
                HumanMessage(content="What are the key applications of AI in healthcare?"),
                AIMessage(content="AI is used in healthcare for medical imaging, drug discovery, personalized treatment, and predictive analytics."),
            ],
            "facts": "The podcast discussed AI applications in healthcare",
            "description": "Test query about last topic"
        },
        {
            "query": "Can you summarize the discussion?",
            "history": [],  # Empty history to test error handling
            "facts": "",
            "description": "Test with empty history and facts"
        }
    ]


class TestPodcastAgent(unittest.TestCase):
    """Test suite for podcast agent functionalities."""
    
    def setUp(self):
        """Set up test environment."""
        logger.info("Setting up test environment")
    
    def tearDown(self):
        """Clean up after tests."""
        logger.info("Cleaning up test environment")
    
    def test_execute_rag_response(self):
        """Test the execute_rag_response function with various inputs."""
        for test_case in TestData.RAG_QUERIES:
            with self.subTest(description=test_case["description"]):
                logger.info(f"Testing RAG response with: {test_case['description']}")
                
                try:
                    # Execute the function
                    result = execute_rag_response(
                        query=test_case["query"],
                        user_intent=test_case["user_intent"],
                        output_emotion=test_case["output_emotion"]
                    )
                    
                    # Check result type
                    self.assertIsInstance(result, test_case["expected_result_type"])
                    
                    # For non-empty queries, check that we got a non-empty response
                    if test_case["query"]:
                        self.assertTrue(len(result) > 0, "Expected non-empty response")
                    
                    logger.info(f"RAG test passed: {test_case['description']}")
                    
                except Exception as e:
                    logger.error(f"RAG test failed: {test_case['description']} - Error: {str(e)}")
                    # Only fail the test if we weren't expecting an error
                    if test_case["query"]:
                        self.fail(f"Unexpected error: {str(e)}")
    
    def test_chatbot_entry(self):
        """Test the chatbot_entry function with various inputs."""
        for test_case in TestData.CHATBOT_TEST_CASES:
            with self.subTest(description=test_case["description"]):
                logger.info(f"Testing chatbot entry with: {test_case['description']}")
                
                try:
                    # Execute the function
                    result = asyncio.run(
                        chatbot_entry(
                            query=test_case["query"],
                            history=test_case["history"],
                            facts=test_case["facts"],
                        )
                    )
                    
                    # Check result type
                    self.assertIsNotNone(result)
                    
                    # Check for expected fields in the result
                    if hasattr(result, "answer"):
                        self.assertIsInstance(result.answer, str)
                        self.assertTrue(len(result.answer) > 0, "Expected non-empty answer")
                    
                    if hasattr(result, "use_rag"):
                        self.assertIsInstance(result.use_rag, bool)
                    
                    if hasattr(result, "user_intent"):
                        self.assertIsInstance(result.user_intent, str)
                    
                    if hasattr(result, "output_emotion"):
                        self.assertIsInstance(result.output_emotion, str)
                    
                    logger.info(f"Chatbot test passed: {test_case['description']}")
                    
                except Exception as e:
                    logger.error(f"Chatbot test failed: {test_case['description']} - Error: {str(e)}")
                    # Only fail the test if we weren't expecting an error
                    if test_case["history"] or test_case["facts"]:
                        self.fail(f"Unexpected error: {str(e)}")


class TestIntegration(unittest.TestCase):
    """Test integration between different components."""
    
    def test_rag_integration_with_chatbot(self):
        """Test integration between RAG and chatbot."""
        logger.info("Testing RAG integration with chatbot")
        
        try:
            # Test query
            query = "What are the latest developments in AI?"
            user_intent = "information"
            output_emotion = "neutral"
            
            # Get RAG response
            rag_result = execute_rag_response(
                query=query,
                user_intent=user_intent,
                output_emotion=output_emotion
            )
            
            # Use RAG result in chatbot
            history = [
                HumanMessage(content=query),
                AIMessage(content=rag_result),
            ]
            
            # Test chatbot with RAG result in history
            chatbot_result = asyncio.run(
                chatbot_entry(
                    query="Can you summarize what you just told me?",
                    history=history,
                    facts=f"The user asked about {query}",
                )
            )
            
            # Verify results
            self.assertIsNotNone(chatbot_result)
            self.assertIsInstance(chatbot_result.answer, str)
            self.assertTrue(len(chatbot_result.answer) > 0)
            
            logger.info("Integration test passed")
            
        except Exception as e:
            logger.error(f"Integration test failed: {str(e)}")
            self.fail(f"Integration test failed: {str(e)}")


def run_tests():
    """Run all tests."""
    logger.info("Starting podcast agent tests")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


if __name__ == "__main__":
    run_tests()

# File: helm/clients/dspy_client.py
from helm.clients.client import Client
from helm.common.cache import CacheConfig
from helm.tokenizers.tokenizer import Tokenizer
from helm.common.cache import Cache
from helm.common.request import Request, RequestResult, GeneratedOutput
import dspy
import os

lm = None
agent = None


class DSPyClient(Client):
    """
    A HELM client that uses DSPy for inference instead of directly calling the model.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        model_name: str = None,
        api_base: str = None,
        api_key: str = None,
    ):
        """
        Initializes the DSPyClient.

        Args:
            tokenizer (Tokenizer): Tokenizer instance (unused but required by HELM interface).
            tokenizer_name (str): Name of the tokenizer (unused but required by HELM interface).
            cache_config (CacheConfig): Configuration for caching.
            model_name (str): The model to use with DSPy.
            api_base (str): Base URL for the model API.
            api_key (str): API key for the DSPy model provider.
        """

        model_name = os.environ.get("DSPY_MODEL_NAME", None)
        api_base = os.environ.get("DSPY_API_BASE", None)
        api_key = os.environ.get("DSPY_API_KEY", None)

        if (model_name == None) or (api_base == None) or (api_key == None):
            raise Exception(
                "\n\nPlease add the model name, api base, and api key for your DSPy program in your environment variables.\n\nexport DSPY_MODEL_NAME=YOUR_MODEL_NAME\nexport DSPY_API_BASE=YOUR_BASE_URL\nexport DSPY_API_KEY=YOUR_API_KEY\n\n"
            )

        global lm, agent
        if lm == None:
            lm = dspy.LM(model=model_name, api_base=api_base, api_key=api_key)
            dspy.configure(lm=lm)
            agent = dspy.Predict("inputs -> answer")

        self.cache = Cache(cache_config) if cache_config else None

    def make_request(self, request: Request) -> RequestResult:
        """
        Handles a request by sending the prompt to DSPy.

        Args:
            request (Request): The request object containing the prompt.

        Returns:
            RequestResult: A HELM-compatible response object.
        """
        prompt_text = request.prompt

        if request.messages:
            prompt_text = " ".join(msg["content"] for msg in request.messages if msg.get("role") != "system")

        try:
            prediction = agent(inputs=prompt_text)
            output_text = prediction.answer if hasattr(prediction, "answer") else str(prediction)
        except Exception as e:
            return RequestResult(success=False, cached=False, completions=[], embedding=[], error=str(e))

        # Return a HELM-compatible RequestResult
        output = GeneratedOutput(text=output_text, logprob=0.0, tokens=[])
        return RequestResult(success=True, cached=False, completions=[output], embedding=[])

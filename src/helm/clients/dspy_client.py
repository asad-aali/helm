# File: helm/clients/dspy_client.py
from helm.clients.client import Client
from helm.common.cache import CacheConfig
from helm.tokenizers.tokenizer import Tokenizer
from helm.common.cache import Cache
from helm.common.request import Request, RequestResult, GeneratedOutput
from helm.proxy.retry import NonRetriableException
import dspy

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

        if not model_name:
            raise NonRetriableException("Please specify the model name in model_deployments.yaml")
        if not api_key:
            raise NonRetriableException("Please provide dspyApiKey key through credentials.conf")

        if ("o3-mini" in model_name) or ("deepseek-r1" in model_name):
            self.lm = dspy.LM(model="openai/" + model_name.split("/")[-1], api_base=api_base, api_key=api_key, temperature=1.0, max_tokens=200000)
        else:
            self.lm = dspy.LM(model="openai/" + model_name.split("/")[-1], api_base=api_base, api_key=api_key)
        
        self.agent = dspy.ChainOfThought("inputs -> output")
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
            with dspy.context(lm=self.lm):
                prediction = self.agent(inputs=prompt_text)
            output_text = prediction.output if hasattr(prediction, "output") else str(prediction)
        except Exception as e:
            return RequestResult(success=False, cached=False, completions=[], embedding=[], error=str(e))

        # Return a HELM-compatible RequestResult
        output = GeneratedOutput(text=output_text, logprob=0.0, tokens=[])
        return RequestResult(success=True, cached=False, completions=[output], embedding=[])
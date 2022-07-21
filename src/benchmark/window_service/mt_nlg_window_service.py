from .gpt2_window_service import GPT2WindowService
from .tokenizer_service import TokenizerService


class MTNLGWindowService(GPT2WindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """
        The max length of the model input. The max sequence length for the MT-NLG models is 2048.
        Source: https://github.com/microsoft/turing-academic-TNLG
        """
        return 2048

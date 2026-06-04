import torch

#from peft import PeftModel
from transformers import PreTrainedModel, AutoConfig, T5ForTokenClassification, AutoModel, AutoTokenizer, AutoModelForTokenClassification

from .configuration_hhem_v2 import HHEMv2Config

class HHEMv2Model(PreTrainedModel):
    config_class = HHEMv2Config

    def __init__(self, config):
        super().__init__(config)
    #     self.t5 = T5ForTokenClassification.from_config(
    #         AutoConfig.from_pretrained(config.foundation)
    #     )

    # def populate(self, model):
    #     self.t5 = model

    # def forward(self, **kwarg):
    #     return self.t5.transformer(**kwarg)

class HHEMv2ForSequenceClassification(PreTrainedModel):
    config_class = HHEMv2Config
    # transformers 5.x accesses all_tied_weights_keys during weight loading;
    # HHEM ties no weights so the correct value is the empty dict.
    all_tied_weights_keys: dict = {}

    def __init__(self, config=HHEMv2Config()):
        super().__init__(config)
        self.t5 = T5ForTokenClassification(
            AutoConfig.from_pretrained(config.foundation)
        )
        self.prompt = config.prompt
        self.tokenzier = AutoTokenizer.from_pretrained(config.foundation)

    def populate(self, model: AutoModel):
        """Initiate the model with the pretrained model

        This method should only be called by Vectara employee who prepares the model for publishing. Users do not need to call this method.

        """
        self.t5 = model

    # TODO: Figure out how to publish only the adapter yet still able to do end-to-end pulling and inference.
    # def populate_lora(self, checkpoint: str):
    #     base_model = AutoModelForTokenClassification.from_pretrained(self.config.foundation)
    #     combined_model = PeftModel.from_pretrained(base_model, checkpoint, is_trainable=False)
    #     self.t5 = combined_model

    def tie_weights(self, **kwargs) -> None:
        # transformers 5.x weight loading replaces tensor objects rather than
        # copying into them, breaking T5's encoder.embed_tokens → shared weight
        # tying that T5ForTokenClassification establishes in its __init__.
        # from_pretrained calls tie_weights(**kwargs) after loading; this override
        # restores the T5-internal tie at that point so predict() scores are
        # correct regardless of how the model is constructed.
        # kwargs (missing_keys, recompute_mapping) apply to the outer model only.
        super().tie_weights(**kwargs)
        self.t5.tie_weights()

    def forward(self, **kwargs): # To cope with `text-classiication` pipeline
        self.t5.eval()
        with torch.no_grad():
            outputs = self.t5(**kwargs)
        logits = outputs.logits
        logits = logits[:, 0, :]
        outputs.logits = logits
        return outputs
        # return self.t5(**kwargs)

    def predict(self, text_pairs):
        tokenizer = self.tokenzier
        pair_dict = [{'text1': pair[0], 'text2': pair[1]} for pair in text_pairs]
        inputs = tokenizer(
            [self.prompt.format(**pair) for pair in pair_dict], return_tensors='pt', padding=True).to(self.t5.device)
        self.t5.eval()
        with torch.no_grad():
            outputs = self.t5(**inputs)
        logits = outputs.logits
        logits = logits[:, 0, :] # tok_cls
        transformed_probs = torch.softmax(logits, dim=-1)
        raw_scores = transformed_probs[:, 1] # the probability of class 1
        return raw_scores



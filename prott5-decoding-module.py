
from transformers import T5ForConditionalGeneration

class ProtT5DecodingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.protT5_model = T5ForConditionalGeneration.from_pretrained("Rostlab/prot_t5_xl_bfd")
        self.protT5_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_bfd")

    def forward(self, latent_repr, max_length=200):
        outputs = self.protT5_model.generate(
            inputs_embeds=latent_repr,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        
        decoded_sequences = self.protT5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_sequences
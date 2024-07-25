from transformers import T5EncoderModel, T5Tokenizer
import torch
import re

class ProtT5EncodingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.protT5_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
        self.protT5_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_bfd")

    def forward(self, sequence):
        processed_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        ids = self.protT5_tokenizer(processed_seq, add_special_tokens=True, return_tensors="pt", padding='longest')
        input_ids = ids['input_ids'].to(self.protT5_model.device)
        attention_mask = ids['attention_mask'].to(self.protT5_model.device)

        with torch.no_grad():
            embedding_repr = self.protT5_model(input_ids=input_ids, attention_mask=attention_mask)
        
        seq_emb = embedding_repr.last_hidden_state
        return seq_emb
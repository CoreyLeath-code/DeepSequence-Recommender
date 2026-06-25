import os
import torch
import numpy as np

class MockSequentialModel(torch.nn.Module):
    """
    A structural representation of a Deep Sequential Recommendation network 
    (such as a Transformer, GRU, or SASRec architecture variant).
    """
    def __init__(self, vocab_size=5000, embedding_dim=128, sequence_length=50):
        super().__init__()
        self.item_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.sequence_layer = torch.nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.scoring_head = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_sequences):
        # input_sequences shape: [Batch Size, Sequence Length]
        embedded = self.item_embeddings(input_sequences)
        gru_out, _ = self.sequence_layer(embedded)
        # Pull down the absolute final hidden sequence state embedding token [Batch Size, Embedding Dim]
        final_state = gru_out[:, -1, :]
        logits = self.scoring_head(final_state)
        return logits

def export_to_onnx_runtime(output_path="models/deep_sequence_rec.onnx"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize the structural network layout
    model = MockSequentialModel()
    model.eval()

    # Generate model input constraints (Batch Size: 1, Evaluation Sequence Window Length: 50)
    dummy_input = torch.randint(0, 5000, (1, 50), dtype=torch.long)

    print(f"🚀 Serializing target Deep Sequence model matrix to: {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_sequences'],
        output_names=['output_logits'],
        dynamic_axes={
            'input_sequences': {0: 'batch_size'},
            'output_logits': {0: 'batch_size'}
        }
    )
    print("✅ Model deployment serialization matrix compilation complete.")

if __name__ == "__main__":
    export_to_onnx_runtime()

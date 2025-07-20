

import torch
import torch.nn as nn
import torch.nn.functional as F


HIDDEN_DIM = 128
class VanillaRNN(nn.Module):
    def __init__(self,embedding_matrix, hidden_dim=HIDDEN_DIM, output_dim=1, freeze_embeddings=True):
        super().__init__()
         # Create the embedding layer using nn.Embedding.from_pretrained
        # It automatically sets num_embeddings and embedding_dim from the matrix shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embeddings)
        self.rnn = nn.RNN(self.embedding.embedding_dim, hidden_dim, batch_first=True) # batch_first=True is important
        self.fc = nn.Linear(hidden_dim, output_dim) # Output dimension for binary classification

    def forward(self, input_ids, attention_mask):
        # 1. Embed the input_ids
        # input_ids shape: (batch_size, sequence_length)
        # embeddings shape: (batch_size, sequence_length, embedding_dim)
        embeddings = self.embedding(input_ids)

        # 2. Get original (non-padded) sequence lengths from attention_mask
        # For each sequence in the batch, sum the '1's in its attention_mask.
        # Ensure lengths are on CPU for pack_padded_sequence if your data is on GPU.
        # This will be a 1D tensor of shape (batch_size,)
        lengths = attention_mask.sum(dim=1).cpu()

        # Handle potential zero-length sequences (though unlikely in sentiment analysis)
        # pack_padded_sequence requires lengths to be > 0.
        # If any length is zero, set it to 1 to avoid error, and mask its contribution later if needed.
        # For typical tokenization, min length will be 2 ([CLS], [SEP]).
        # If there's a risk of 0 lengths, you might replace them with 1 and then zero out their contribution to loss.
        # For most practical cases with a tokenizer adding special tokens, this isn't strictly necessary
        # but good to be aware of.
        lengths = torch.max(lengths, torch.tensor(1, device=lengths.device))

        # 3. Pack the padded sequence
        # This prevents the RNN from processing padding tokens.
        # It creates a PackedSequence object that the RNN understands.
        # embeddings shape: (batch_size, sequence_length, embedding_dim)
        # lengths shape: (batch_size,)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False # enforce_sorted=False if your lengths aren't sorted
        )

        # 4. Pass the packed sequence through the RNN
        # hidden_states: PackedSequence object containing output for non-padded tokens
        # final_hidden_state: Tensor containing the last hidden state for each sequence (h_n, c_n for LSTM)
        # Note: for RNN, final_hidden_state is (num_layers * num_directions, batch_size, hidden_size)
        # For a simple, non-bidirectional RNN with 1 layer, this is (1, batch_size, hidden_size)
        _, final_hidden_state = self.rnn(packed_embeddings)

        # 5. Unpack the hidden states (optional, but useful if you plan to use attention on all hidden states)
        # This converts the PackedSequence back to a padded tensor with original batch_first structure.
        # padded_hidden_states, _ = nn.utils.rnn.pad_packed_sequence(
        #     packed_hidden_states, batch_first=True, total_length=input_ids.shape[1]
        # )
        # padded_hidden_states shape: (batch_size, sequence_length, hidden_dim)

        # 6. Use the final hidden state for classification
        # For a simple RNN, the final_hidden_state (h_n) represents the context of the sequence.
        # Since batch_first=True in the RNN, final_hidden_state is (1, batch_size, hidden_dim).
        # We need to remove the first dimension (num_layers * num_directions) to get (batch_size, hidden_dim).
        # If it were bidirectional, final_hidden_state would be (2, batch_size, hidden_dim),
        # requiring concatenation of forward/backward states before the fc layer.
        final_state_squeezed = final_hidden_state.squeeze(0) # shape: (batch_size, hidden_dim)

        # 7. Pass through the final linear layer
        # output will be logits, shape: (batch_size, output_dim)
        output = self.fc(final_state_squeezed)

        return output



class VanillaLSTM(nn.Module):
    def __init__(self,embedding_matrix, hidden_dim=HIDDEN_DIM, output_dim=1,freeze_embeddings=True):
        super().__init__()
        self.embedding =  nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embeddings)
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_dim, batch_first=True) # batch_first=True is important
        self.fc = nn.Linear(hidden_dim, output_dim) # Output dimension for binary classification

    def forward(self, input_ids, attention_mask):
        # 1. Embed the input_ids
        embeddings = self.embedding(input_ids)

        # 2. Get original (non-padded) sequence lengths from attention_mask
        lengths = attention_mask.sum(dim=1).cpu()

        # 3. Pack the padded sequence
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )

        # 4. Pass the packed sequence through the LSTM
        _, (final_hidden_state, _) = self.lstm(packed_embeddings)

        # 5. Use the final hidden state for classification
        final_state_squeezed = final_hidden_state.squeeze(0) # shape: (batch_size, hidden_dim)

        # 6. Pass through the final linear layer
        output = self.fc(final_state_squeezed)

        return output
    
class VanillaBidirectionalRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=HIDDEN_DIM, output_dim=1,freeze_embeddings=True):
        super().__init__()
        self.embedding =  nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embeddings)
        self.rnn = nn.RNN(self.embedding.embedding_dim, hidden_dim, batch_first=True, bidirectional=True) # bidirectional=True
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # Output dimension for binary classification

    def forward(self, input_ids, attention_mask):
        # 1. Embed the input_ids
        embeddings = self.embedding(input_ids)

        # 2. Get original (non-padded) sequence lengths from attention_mask
        lengths = attention_mask.sum(dim=1).cpu()

        # 3. Pack the padded sequence
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )

        # 4. Pass the packed sequence through the RNN
        _, final_hidden_state = self.rnn(packed_embeddings)

        # 5. Use the final hidden state for classification
        # For bidirectional RNNs, final_hidden_state will have shape (2, batch_size, hidden_dim)
        final_state_squeezed = final_hidden_state.view(final_hidden_state.size(1), -1) # shape: (batch_size, hidden_dim * 2)

        # 6. Pass through the final linear layer
        output = self.fc(final_state_squeezed)  




        return output
    


class VanillaBidirectionalLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=HIDDEN_DIM, output_dim=1, freeze_embeddings=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze_embeddings)  # Freeze embeddings by default
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_dim, batch_first=True, bidirectional=True) # bidirectional=True
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # Output dimension for binary classification

    def forward(self, input_ids, attention_mask):
        # 1. Embed the input_ids
        embeddings = self.embedding(input_ids)

        # 2. Get original (non-padded) sequence lengths from attention_mask
        lengths = attention_mask.sum(dim=1).cpu()

        # 3. Pack the padded sequence
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )

        # 4. Pass the packed sequence through the LSTM
        _, (final_hidden_state, _) = self.lstm(packed_embeddings)

        # 5. Use the final hidden state for classification
        final_state_squeezed = final_hidden_state.view(final_hidden_state.size(1), -1) # shape: (batch_size, hidden_dim * 2)

        # 6. Pass through the final linear layer
        output = self.fc(final_state_squeezed)                              

        return output









import torch
import torch.nn as nn
import torch.nn.functional as F


## Assuming hidden_dim is the size of the hidden state of the RNN/encoder
class BahdanauAttention(nn.Module):
    def __init__(self, decoder_hidden_dim, encoder_hidden_dim,attention_hidden_dim):
        super(BahdanauAttention, self).__init__()
        # W_1 (also sometimes called W_query or W_decoder)
        self.W1 = nn.Linear(decoder_hidden_dim,attention_hidden_dim, bias=False) ## d_decoder * d_attention
        # W_2 (also sometimes called W_key or W_encoder)
        self.W2 = nn.Linear(encoder_hidden_dim, attention_hidden_dim, bias=False) ## d_encoder * d_attention
        # v_a (also sometimes called V or W_score)
        self.V = nn.Linear(attention_hidden_dim, 1, bias=False)   ## d_attention * 1     

    def forward(self, query_state, encoder_hidden_states, attention_mask=None):
        # query_state: (batch_size, hidden_dim) - this is typically the decoder's current hidden state
        # encoder_hidden_states: (batch_size, seq_len, hidden_dim) - these are the outputs from the encoder

        # Step 1 & 2: Transform the query and keys
        # transformed_query: (batch_size, hidden_dim)
        transformed_query = self.W1(query_state)
        # transformed_encoder_states: (batch_size, seq_len, hidden_dim)
        transformed_encoder_states = self.W2(encoder_hidden_states)

        # Step 3: Combine (Add) and Apply Non-Linearity
        # We need to unsqueeze transformed_query to (batch_size, 1, hidden_dim)
        # so it can be broadcasted and added to transformed_encoder_states.
        # energies: (batch_size, seq_len, hidden_dim)
        energies = torch.tanh(transformed_query.unsqueeze(1) + transformed_encoder_states)

        # Step 4: Calculate Raw Attention Scores (Energies)
        # attention_scores: (batch_size, seq_len, 1)
        attention_scores = self.V(energies).squeeze(2) # squeeze to (batch_size, seq_len)

        # Apply attention mask if provided (important for padded sequences!)
        if attention_mask is not None:
            # Mask out padded positions. Fill with a very small negative number
            # so that after softmax, they become virtually 0.
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        # Step 5: Compute Attention Weights
        # attention_weights: (batch_size, seq_len) - sum to 1 across seq_len dim
        attention_weights = F.softmax(attention_scores, dim=1)

        # Step 6: Calculate Context Vector
       
        # context_vector: (batch_size, hidden_dim)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_hidden_states).squeeze(1)
        
        return context_vector, attention_weights
    




class LuongDotProductAttention(nn.Module):
    def __init__(self,decoder_hidden_dim, encoder_hidden_dim):
        super(LuongDotProductAttention, self).__init__()

        if decoder_hidden_dim != encoder_hidden_dim:
            raise ValueError("Decoder hidden dimension must match encoder hidden dimension for Luong dot product attention.")
        # W_1 (also sometimes called W_query or W_decoder)

    def forward(self,query_state, encoder_hidden_states,attention_mask = None):

        # (B,D) -> (B,D,1)
        # (B,L,D) @ (B,D,1) -> (B,L,1)

        attention_scores = torch.bmm(encoder_hidden_states,query_state.unsqueeze(2)).squeeze(2)

        if attention_mask is not None:
            # Mask out padded positions. Fill with a very small negative number
            # so that after softmax, they become virtually 0.
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=1)


        #  (B,L) => (B,1,L) @ (B,L,D) => (B,1,D) =>(B,D)

        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_hidden_states).squeeze(1)
       

        return context_vector, attention_weights
    


class LuongGeneralAttention(nn.Module):
    def __init__(self,decoder_hidden_dim, encoder_hidden_dim):
        super(LuongGeneralAttention, self).__init__()

     
        # W_a
        self.W_a = nn.Linear(decoder_hidden_dim,encoder_hidden_dim ,bias=False)  ### d_h  -->d_e 
   


    def forward(self,query_state, encoder_hidden_states,attention_mask = None):

        # (B,D) -> (B,D,1)
        # (B,L,D) @ (B,D,1) -> (B,L,1)
        query_state_transformed = self.W_a(query_state)

        attention_scores = torch.bmm(encoder_hidden_states,query_state_transformed.unsqueeze(2)).squeeze(2)






        if attention_mask is not None:
            # Mask out padded positions. Fill with a very small negative number
            # so that after softmax, they become virtually 0.
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=1)


        #  (B,L) => (B,1,L) @ (B,L,D) => (B,1,D) =>(B,D)

        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_hidden_states).squeeze(1)
       

        return context_vector, attention_weights
    


class LuongConcatAttention(nn.Module):
    def __init__(self,decoder_hidden_dim, encoder_hidden_dim,attention_hidden_dim):
        super(LuongConcatAttention, self).__init__()

     
        # W_c
        self.W_c = nn.Linear(decoder_hidden_dim + encoder_hidden_dim,attention_hidden_dim, bias=False)

        self.V = nn.Linear(attention_hidden_dim, 1, bias=False)   ## d_attention * 1    


    def forward(self,query_state, encoder_hidden_states,attention_mask = None):

        # (B,D_d) => (B,1,D_d) => (B,L,D_d)
        transformed_query = query_state.unsqueeze(1).expand(-1, encoder_hidden_states.size(1), -1)  # (B, L, D_d)





        concat_query_encoder_hidden_states = torch.cat((transformed_query,encoder_hidden_states),dim=2)  #(B, L, D_d + D_e)



        energies = torch.tanh(self.W_c(concat_query_encoder_hidden_states))#(B,L,D_a)


    
        # attention_scores: (batch_size, seq_len, 1)
        attention_scores = self.V(energies).squeeze(2) # squeeze to (batch_size, seq_len)

        if attention_mask is not None:
            # Mask out padded positions. Fill with a very small negative number
            # so that after softmax, they become virtually 0.
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=1)


        #  (B,L) => (B,1,L) @ (B,L,D) => (B,1,D) =>(B,D)

        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_hidden_states).squeeze(1)
       

        return context_vector, attention_weights
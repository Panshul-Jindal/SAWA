from attention_classes import BahdanauAttention,LuongDotProductAttention,LuongGeneralAttention,LuongConcatAttention
import torch
import torch.nn as nn   
import torch.nn.functional as F


HIDDEN_DIM = 128
# --- General Purpose RNN Model ---
class VanillaRNNWithPlugAndPlayAttention(nn.Module):
    def __init__(self,embedding_matrix, attention_class, attention_kwargs,return_attention_weights =False,
                  hidden_dim=HIDDEN_DIM, output_dim=1,freeze_embeddings=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,freeze= freeze_embeddings)
           # We need to ensure the RNN outputs all hidden states. batch_first=True is important.
        self.rnn = nn.RNN(self.embedding.embedding_dim, hidden_dim, batch_first=True)
        
        # Instantiate the attention module using the provided class and kwargs
        # The **attention_kwargs unpacks the dictionary into named arguments
        self.attention = attention_class(
            decoder_hidden_dim=hidden_dim, # Decoder's hidden state is the query for attention
            encoder_hidden_dim=hidden_dim, # Encoder's outputs are the keys/values for attention
            **attention_kwargs # This will pass specific kwargs like 'attention_dim' if present
        )
        self.return_attention_weights = return_attention_weights
  


        # The final fully connected layer will now take the concatenated 
        # final hidden state AND the context vector from attention.
        # So, its input dimension becomes hidden_dim + hidden_dim = 2 * hidden_dim

        # Assuming context_vector will also be of hidden_dim size (typical)   (! I don't find where it isn't the case!)

        self.fc = nn.Linear(hidden_dim * 2, output_dim) 

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
        packed_output, final_hidden_state = self.rnn(packed_embeddings)

      # 5. Unpack the RNN outputs to get all encoder_hidden_states

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=input_ids.size(1)
        )
          # 6 Squeeze the final_hidden_state to (batch_size, hidden_dim) to use as query
        query = final_hidden_state.squeeze(0) 

        # Calculate context vector using the plug-and-play attention module
        context_vector, attention_weights = self.attention(
            query_state=query, 
            encoder_hidden_states=outputs, 
            attention_mask=attention_mask
        )
 # 7. Concatenate the query (final hidden state) with the context vector
        # This combines the "sequential summary" with the "attention-weighted summary"
        combined_representation = torch.cat((query, context_vector), dim=1)
        output = self.fc(combined_representation)  

        if self.return_attention_weights:
            # If we want to return attention weights for visualization/debugging
            return output, attention_weights
        else:
            return output 




class VanillaBidirectionalRNNWithPlugAndPlayAttention(nn.Module):
    def __init__(self,embedding_matrix, attention_class, attention_kwargs,return_attention_weights =False,
                 hidden_dim=HIDDEN_DIM, output_dim=1,freeze_embeddings=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,freeze= freeze_embeddings)
        self.rnn = nn.RNN(self.embedding.embedding_dim, hidden_dim, batch_first=True, bidirectional=True) # bidirectional=True
        self.fc = nn.Linear(hidden_dim * 4, output_dim) # Output dimension for binary classification


        self.attention = attention_class(
            decoder_hidden_dim=2*hidden_dim, # Decoder's hidden state is the query for attention
            encoder_hidden_dim=2*hidden_dim, # Encoder's outputs are the keys/values for attention
            **attention_kwargs # This will pass specific kwargs like 'attention_dim' if present
        )
        self.return_attention_weights = return_attention_weights


    def forward(self, input_ids, attention_mask):
        # 1. Embed the input_ids
        embeddings = self.embedding(input_ids) #(B,L,E)

        # 2. Get original (non-padded) sequence lengths from attention_mask
        lengths = attention_mask.sum(dim=1).cpu()

        # 3. Pack the padded sequence
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )

        # 4. Pass the packed sequence through the RNN
        # packed_hidden_states: PackedSequence of (B, L, 2*H)
        # final_hidden_state: (num_layers * num_directions, B, H) -> (2, B, H) for 1-layer bidirectional
        packed_hidden_states, final_hidden_state = self.rnn(packed_embeddings)
            
        # 5. Unpack the rnn outputs to get all encoder_hidden_states
        # outputs: (B, L, 2*H) - this contains the concatenated forward and backward hidden states for each time step

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_hidden_states, batch_first=True, total_length=input_ids.size(1)
        )
        # For bidirectional RNNs, final_hidden_state will have shape (2, batch_size, hidden_dim)
        #6 Squeeze the final_hidden_state  (2,batch_size,hidden_dim) to (batch_size, 2*hidden_dim) to use as query
        query = final_hidden_state.view(final_hidden_state.size(1), -1) # shape: (batch_size, hidden_dim * 2)
        #7Calculate context vector using the plug-and-play attention module
        context_vector, attention_weights = self.attention(
            query_state=query, 
            encoder_hidden_states=outputs, 
            attention_mask=attention_mask
        )

    
       


        # 8. Concatenate the query (final hidden state) with the context vector
        # query: (B, 2*H)
        # context_vector: (B, 2*H) - typically, attention outputs context vector of encoder_hidden_dim
        # combined_representation: (B, 4*H)
        # This combines the "sequential summary" with the "attention-weighted summary"


        combined_representation = torch.cat((query, context_vector), dim=1)    #(B,2*H)  +(B,2*hidden_dim) = (B,4*hidden_dim)
        

      
        # 9. Pass through the final linear layer
        output = self.fc(combined_representation)  



 # Return attention weights for visualization/debugging
  
        if self.return_attention_weights:
        
            return output, attention_weights
        else:
            return output  # Only return the output if weights are not explicitly requested



# --- General Purpose LSTM Model ---
class VanillaLSTMWithPlugAndPlayAttention(nn.Module):
    def __init__(self,embedding_matrix, attention_class, attention_kwargs,return_attention_weights =False,
                 hidden_dim=HIDDEN_DIM, output_dim=1,freeze_embeddings=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,freeze= freeze_embeddings)
           # We need to ensure the RNN outputs all hidden states. batch_first=True is important.
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_dim, batch_first=True)
        
        # Instantiate the attention module using the provided class and kwargs
        # The **attention_kwargs unpacks the dictionary into named arguments
        self.attention = attention_class(
            decoder_hidden_dim=hidden_dim, # Decoder's hidden state is the query for attention
            encoder_hidden_dim=hidden_dim, # Encoder's outputs are the keys/values for attention
            **attention_kwargs # This will pass specific kwargs like 'attention_dim' if present
        )
        self.return_attention_weights = return_attention_weights
  


        # The final fully connected layer will now take the concatenated 
        # final hidden state AND the context vector from attention.
        # So, its input dimension becomes hidden_dim + hidden_dim = 2 * hidden_dim

        # Assuming context_vector will also be of hidden_dim size (typical)   (! I don't find where it isn't the case!)

        self.fc = nn.Linear(hidden_dim * 2, output_dim) 

    def forward(self, input_ids, attention_mask):
                # 1. Embed the input_ids
        embeddings = self.embedding(input_ids)

                # 2. Get original (non-padded) sequence lengths from attention_mask
        lengths = attention_mask.sum(dim=1).cpu()
   # 3. Pack the padded sequence
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
     
        # 4. Pass the packed sequence through the lstm
        packed_output, (final_hidden_state, _)  = self.lstm(packed_embeddings)

      # 5. Unpack the LSTM outputs to get all encoder_hidden_states

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=input_ids.size(1)
        )
          # 6 Squeeze the final_hidden_state to (batch_size, hidden_dim) to use as query
        query = final_hidden_state.squeeze(0) 

        # Calculate context vector using the plug-and-play attention module
        context_vector, attention_weights = self.attention(
            query_state=query, 
            encoder_hidden_states=outputs, 
            attention_mask=attention_mask
        )
 # 7. Concatenate the query (final hidden state) with the context vector
        # This combines the "sequential summary" with the "attention-weighted summary"
        combined_representation = torch.cat((query, context_vector), dim=1)
        output = self.fc(combined_representation)  


        # Conditionally return attention weights on basis of flags

        # Allows for visualization or debugging of attention mechanisms
        if self.return_attention_weights:
            return output, attention_weights
        else:
            return output 



## General Purpose Bidirectional LSTM Model
class VanillaBidirectionalLSTMWithPlugAndPlayAttention(nn.Module):
    def __init__(self,embedding_matrix, attention_class, attention_kwargs,return_attention_weights =False,
                 hidden_dim=HIDDEN_DIM, output_dim=1,freeze_embeddings=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,freeze= freeze_embeddings)
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_dim, batch_first=True, bidirectional=True) # bidirectional=True
        self.fc = nn.Linear(hidden_dim * 4, output_dim) # Output dimension for binary classification


        self.attention = attention_class(
            decoder_hidden_dim=2*hidden_dim, # Decoder's hidden state is the query for attention
            encoder_hidden_dim=2*hidden_dim, # Encoder's outputs are the keys/values for attention
            **attention_kwargs # This will pass specific kwargs like 'attention_dim' if present
        )
        self.return_attention_weights = return_attention_weights


    def forward(self, input_ids, attention_mask):
        # 1. Embed the input_ids
        embeddings = self.embedding(input_ids) #(B,L,E)

        # 2. Get original (non-padded) sequence lengths from attention_mask
        lengths = attention_mask.sum(dim=1).cpu()

        # 3. Pack the padded sequence
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )

        # 4. Pass the packed sequence through the LSTM
        # packed_hidden_states: PackedSequence of (B, L, 2*H)
        # final_hidden_state: (num_layers * num_directions, B, H) -> (2, B, H) for 1-layer bidirectional
        packed_hidden_states, (final_hidden_state, _) = self.lstm(packed_embeddings)
            
        # 5. Unpack the rnn outputs to get all encoder_hidden_states
        # outputs: (B, L, 2*H) - this contains the concatenated forward and backward hidden states for each time step

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_hidden_states, batch_first=True, total_length=input_ids.size(1)
        )
        # For bidirectional LSTMs, final_hidden_state will have shape (2, batch_size, hidden_dim)
        #6 Squeeze the final_hidden_state  (2,batch_size,hidden_dim) to (batch_size, 2*hidden_dim) to use as query
        query = final_hidden_state.view(final_hidden_state.size(1), -1) # shape: (batch_size, hidden_dim * 2)
        #7Calculate context vector using the plug-and-play attention module
        context_vector, attention_weights = self.attention(
            query_state=query, 
            encoder_hidden_states=outputs, 
            attention_mask=attention_mask
        )

    
       


        # 8. Concatenate the query (final hidden state) with the context vector
        # query: (B, 2*H)
        # context_vector: (B, 2*H) - typically, attention outputs context vector of encoder_hidden_dim
        # combined_representation: (B, 4*H)
        # This combines the "sequential summary" with the "attention-weighted summary"


        combined_representation = torch.cat((query, context_vector), dim=1)    #(B,2*H)  +(B,2*hidden_dim) = (B,4*hidden_dim)
        

      
        # 9. Pass through the final linear layer
        output = self.fc(combined_representation)  

 # Return attention weights for visualization/debugging
  
        if self.return_attention_weights:
        
            return output, attention_weights
        else:
            return output  # Only return the output if weights are not explicitly requested

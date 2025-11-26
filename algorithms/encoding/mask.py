import torch

# 1 means not masked, 0 means masked
# (batch, seq_len) is input
# (batch, heads, Q_seq_len, K_seq_len) is the attention score to be masked

def create_mha_padding_mask(input_ids, padding_ids=torch.tensor([0])):
    padding_ids = padding_ids.to(input_ids.device)

    # (batch, seq_len) -> (batch, seq_len)
    mask = ~torch.any(input_ids.unsqueeze(-1) == padding_ids, dim=-1)
    # (batch, 1, 1, seq_len)
    mask = mask.unsqueeze(1).unsqueeze(1)

    return mask

def create_mha_causal_mask(input_ids):
    seq_len = input_ids.size(1)

    '''
    [0, 1, 1, 1]
    [0, 0, 1, 1]
    [0, 0, 0, 1]
    [0, 0, 0, 0]
    '''
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = ~mask.unsqueeze(0).unsqueeze(0)
    mask = mask.to('cuda')

    '''
    [[
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
    ]]
    '''

    return mask

# to create a combined mask, & the two masks together

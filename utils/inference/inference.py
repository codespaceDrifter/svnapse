import torch

def inference(model, tokenizer, max_length=200):
    model.eval()
    model.cuda()  # Move model to CUDA
    
    print("Interactive inference mode. Type 'exit' to quit.")
    
    while True:
        user_input = input("Enter text: ")
        
        if user_input.lower() == 'exit':
            break
        
        x = tokenizer.encode(user_input, add_SOS=True, add_EOS=False).unsqueeze(0).cuda()
        y = tokenizer.encode("", add_SOS=True, add_EOS=False).unsqueeze(0).cuda()
        with torch.no_grad():
            while y.size(-1) < max_length:
                # (batch, 1)
                next_token_ids = model.predict(x, y)

                if torch.all(next_token_ids == tokenizer.EOS_ID):
                    break
                next_token_word = tokenizer.decode(next_token_ids.cpu())
                print(next_token_word[0], end=" ", flush=True)

                y = torch.cat((y, next_token_ids), dim=1)

        print()  # Add newline after generation is complete

            

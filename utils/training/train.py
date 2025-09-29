import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from ..model.tokenizer import Tokenizer
from .saves import cleanup_checkpoints, load_latest_checkpoint
import numpy as np


#TO DO: differentiate the core training loop of input output loss as a function to pass in
# dataset, dataloader, save and load, print loss, are all the same for all models
'''
            outputs = model(inputs,targets[:, :-1])
            outputs = outputs.permute(0, 2, 1)
            loss = loss_fn(outputs, targets[:, 1:].long())
these 3 lines basically. the forward_pass function

'''

def train_model(
    train_dataset,
    test_dataset,
    model,
    optimizer,
    batch_size,
    save_folder_path,
    num_epochs,
    loss_fn,
    tokenizer = None,
    clip_grad_norm = 2.0,
    batch_per_save = 10
):
    model.cuda()
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Arrays to store losses for plotting
    train_losses = []
    test_losses = []
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_folder_path), exist_ok=True)
    
    # Try to load latest checkpoint and get last epoch/batch
    last_epoch, last_batch = load_latest_checkpoint(save_folder_path, model)

    batch_idx = last_batch
    
    # Training loop
    for epoch in range(last_epoch, num_epochs):
        model.train()
        
        print(f"\nEpoch {epoch}/{num_epochs}")
        # Skip entire epoch if we've already trained it
            
        for inputs, targets in train_loader:

            batch_idx += 1
                
            inputs = inputs.cuda()
            targets = targets.cuda()
   
            loss = model.compute_loss(inputs, targets)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            
            # Print loss every 10 batches
            if (batch_idx) % batch_per_save == 0:
                current_loss = loss.item()
                train_losses.append(current_loss)
                print(f"Batch {batch_idx}, Train Loss: {current_loss:.4f}")

            # save every 10 batches
            if (batch_idx) % batch_per_save == 0:
                state = model.state_dict()
                torch.save(state, f"{save_folder_path}/epoch_{epoch}_batch_{batch_idx}.pt")
                cleanup_checkpoints(save_folder_path)

            # eval one test batch every 10 batches
            if (batch_idx) % batch_per_save == 0:
                model.eval()
                with torch.no_grad():
                    try:
                        test_inputs, test_targets = next(iter(test_loader))
                        test_inputs = test_inputs.cuda()
                        test_targets = test_targets.cuda()


                        test_outputs = model(test_inputs, test_targets[:, :-1])
                        permuted_outputs = test_outputs.permute(0, 2, 1)
                        test_loss = loss_fn(permuted_outputs, test_targets[:, 1:].long()).item()
                        test_losses.append(test_loss)
                        print(f"Test Loss: {test_loss:.4f}")

                        print("Target:")
                        target_text = ' '.join(tokenizer.decode(test_targets.cpu().numpy()))
                        print(target_text)
                        
                        print("Output:")
                        pred_id = test_outputs.argmax(dim=-1)
                        output_text = ' '.join(tokenizer.decode(pred_id.cpu().numpy()))
                        print(output_text)

                    except StopIteration:
                        print("No more test batches")
                        break
        
    # Plot the losses
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Batch (x10)')
    plt.ylabel('Loss')
    
    # Plot test loss
    plt.subplot(1, 2, 2)
    batch_indices = np.linspace(0, len(train_losses)-1, len(test_losses))
    plt.plot(batch_indices, test_losses, 'r')
    plt.title('Test Loss')
    plt.xlabel('Equivalent Batch (x10)')
    plt.ylabel('Loss')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f"{save_folder_path}/loss_plot.png")
    plt.show()
    
    return train_losses, test_losses
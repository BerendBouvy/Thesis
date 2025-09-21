from tqdm import tqdm
import torch



def train(model, dataloader, optimizer, prev_updates, device, batch_size, writer=None, verbose=True):
    """
    Trains the model on the given data.
    
    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The data loader.
        loss_fn: The loss function.
        optimizer: The optimizer.
    """
        
    model.train()  # Set the model to training mode

    for batch_idx, data in enumerate(tqdm(dataloader, disable=not verbose)):
        n_upd = prev_updates + batch_idx
        
        dataX = data[0].to(device)
        datay = data[1].to(device)
        
        optimizer.zero_grad()  # Zero the gradients

        output = model(dataX, datay)  # Forward pass
        loss = output.loss
        
        loss.backward()
        
        if n_upd % 100 == 0:
            # Calculate and log gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            if verbose:
                print(f'Step {n_upd:,} (N samples: {n_upd*batch_size:,}), Loss: {loss.item():.4f} (Recon: {output.loss_recon.item():.4f}, KL: {output.loss_kl.item():.4f}) Grad: {total_norm:.4f}')

        if writer is not None:
            global_step = n_upd
            writer.add_scalar('Loss/Train', loss.item(), global_step)
            writer.add_scalar('Loss/Train/recon', output.loss_recon.item(), global_step)
            writer.add_scalar('Loss/Train/KLD', output.loss_kl.item(), global_step)
            # writer.add_scalar('GradNorm/Train', total_norm, global_step)

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    
        
        optimizer.step()  # Update the model parameters
        
    return prev_updates + len(dataloader)
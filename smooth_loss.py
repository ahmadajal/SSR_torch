import torch
import numpy as np

EPS = 1e-12
config=dict(device="cuda")

def BetaLoss(inputs, logit, alpha=0.3, scale=1.0):
    W = compute_W(inputs, logit)

    def get_smooth_loss(y_pred):
        A = torch.diag_embed(y_pred)  # (B, cl, cl)
        p = torch.unsqueeze(y_pred, 2) # (B, cl, 1)
        A = A - torch.matmul(p, torch.transpose(p, 1, 2)) # A-pp^T
        A = A + torch.eye(int(A.size()[1]), int(A.size()[2])).unsqueeze(0).to(config["device"]) * EPS
        e, U = torch.linalg.eigh(A)
        e = torch.clamp(e, min=1e-20, max=np.inf)
        Sigma = torch.diag_embed(torch.sqrt(e))
        L = torch.matmul(U, Sigma)
        B = torch.matmul(W, L)  # (B, K, cl) x (B, cl, cl) --> (B, K, cl)
        B_T = torch.transpose(B, 1, 2) # (B, cl, K)
        C = torch.matmul(B_T, B) # (B, cl, cl)
        C = C + torch.eye(int(C.size()[1]), int(C.size()[2])).unsqueeze(0).to(config["device"]) * EPS
        H_e, _ = torch.linalg.eigh(C)
        avg_beta = torch.mean(H_e[:, -1]) # Eigenvalues. Shape is [..., N]. Sorted in non-decreasing order.
        return avg_beta * scale


    def smoothCE_loss(y_pred, y_true):
        criterion = torch.nn.CrossEntropyLoss()
        target_loss = criterion(y_pred, y_true)
        smoothness = get_smooth_loss(y_pred)
        return alpha * smoothness + (1-alpha) * target_loss

    return smoothCE_loss


def compute_W(inputs, logits):
    C = logits.size()[1]
    ws = []
    if inputs.size()[0] > 1:
        inputs = (inputs,)
    with torch.autograd.set_grad_enabled(True):
        for c in range(C):
            w = torch.autograd.grad(torch.unbind(logits[:, c]), inputs, create_graph=True)[0]
            w = torch.mean(w, dim=-1)
            w = w.flatten(start_dim=1)
            # w = w.unsqueeze(-1)
            ws.append(w)
    return torch.stack(ws, dim=-1)

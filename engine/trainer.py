import torch, torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, lr=5e-4, wd=5e-3, patience=10, logdir='runs'):
        self.model, self.patience = model, patience
        self.crit = nn.CrossEntropyLoss()
        self.opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
        self.tb = SummaryWriter(logdir)

    def fit(self, train_loader, val_loader, epochs=50, device='cuda'):
        m, best, wait = self.model.to(device), 0, 0
        best_w = m.state_dict()
        for ep in range(epochs):
            m.train(); loss_epoch = 0
            for i, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)
                self.opt.zero_grad()
                out = m(xb)
                loss = self.crit(out, yb)
                loss.backward()
                self.opt.step()
                loss_epoch += loss.item()
            acc = self.evaluate(val_loader, device)
            print(f"[Epoch {ep+1}/{epochs}] loss={loss_epoch:.4f} val_acc={acc:.4f}")  # 👈 可视化打印
            self.tb.add_scalar('val/acc', acc, ep)
            if acc > best:
                best, wait, best_w = acc, 0, m.state_dict()
            else:
                wait += 1
            if wait >= self.patience:
                break
        m.load_state_dict(best_w)
        return best

    @torch.no_grad()
    def evaluate(self, loader, device='cuda'):
        self.model.eval(); c, t = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            c += (self.model(xb).argmax(1) == yb).sum().item()
            t += len(yb)
        return c / t if t else 0

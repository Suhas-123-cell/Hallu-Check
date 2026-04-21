import torch
x = torch.tensor([1.0], requires_grad=True)
y = x * 2

def dummy_checkpoint(func, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(func, *args, **kwargs)

def my_func(a):
    return a * a

out = dummy_checkpoint(my_func, y)
loss = out.sum()
loss.backward()
print("Success checkpoint")

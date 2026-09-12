"""Correct censored Weibull likelihood; separate from archived original reproduction."""
import torch

def weibull_nll(scale,shape,time,event,reduction='mean'):
    if not torch.all((scale>0)&(shape>0)&(time>0)):
        raise ValueError('scale, shape and time must be positive')
    if not torch.all((event==0)|(event==1)):
        raise ValueError('event must be binary')
    log_ratio=torch.log(time)-torch.log(scale)
    cumulative=torch.exp(shape*log_ratio)
    log_hazard=torch.log(shape)-torch.log(scale)+(shape-1)*log_ratio
    values=cumulative-event*log_hazard
    if reduction=='none': return values
    if reduction=='mean': return values.mean()
    raise ValueError('reduction must be mean or none')

if __name__=='__main__':
    scale=torch.tensor([2.,3.],dtype=torch.float64,requires_grad=True)
    shape=torch.tensor([1.,2.],dtype=torch.float64,requires_grad=True)
    time=torch.tensor([4.,2.],dtype=torch.float64); event=torch.tensor([1.,0.],dtype=torch.float64)
    dist=torch.distributions.Weibull(scale,shape)
    expected=(-event*dist.log_prob(time)+(1-event)*(time/scale)**shape).mean()
    actual=weibull_nll(scale,shape,time,event)
    assert torch.allclose(actual,expected)
    actual.backward(); assert torch.isfinite(scale.grad).all() and torch.isfinite(shape.grad).all()
    assert torch.autograd.gradcheck(lambda a,b:weibull_nll(a,b,time,event),(scale,shape))
    print('PASS: Weibull density/censoring agreement and gradient check')

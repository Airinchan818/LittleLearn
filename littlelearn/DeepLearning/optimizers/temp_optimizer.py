import jax.numpy as jnp
from littlelearn.DeepLearning.layers import Parameter


class Optimizer:
    def __init__(self):
        self.lr = None
        self.iterator = 0

    def step(self):
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, beta=(0.9, 0.999), eps=1e-7,clip_norm :float = 0.0):
        super().__init__()

        for p in params:
            if not isinstance(p, Parameter):
                raise TypeError("All elements in params must be Parameter objects")

        self.params = params
        self.lr = lr
        self.beta1 = beta[0]
        self.beta2 = beta[1]
        self.eps = eps
        self.clipnorm = clip_norm

        
        self.m = [jnp.zeros_like(p.tensor) for p in params]
        self.v = [jnp.zeros_like(p.tensor) for p in params]

    def step(self):
        t = self.iterator 

        new_m = []
        new_v = []

        for i, param in enumerate(self.params):
            g = param.grad
            if g is None:
                new_m.append(self.m[i])
                new_v.append(self.v[i])
                continue
            if self.clipnorm > 0.0:
                norm = jnp.linalg.norm(g)
                if norm > self.clipnorm :
                    scale = self.clipnorm / norm
                    g = g * scale


            m = self.beta1 * self.m[i] + (1 - self.beta1) * g


            v = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            m_hat = m / (1 - self.beta1 ** (t + 1))
            v_hat = v / (1 - self.beta2 ** (t + 1))


            update = self.lr / (jnp.sqrt(v_hat) + self.eps) * m_hat


            param.tensor = param.tensor - update

            new_m.append(m)
            new_v.append(v)

        self.m = new_m
        self.v = new_v

        self.iterator =+1 


class AdamW (Optimizer) :
    def __init__(self,param : list,lr = 1e-3,beta=(0.9,0.999),decay=4e-3,epsilon=1e-7,clip_norm : float = 0.0) :
        super().__init__()
        for p in param :
            if not isinstance(p,Parameter) :
                raise TypeError("All elements in params must be Parameter objects")
        self.params = param 
        self.lr = lr 
        self.beta1 = beta[0]
        self.beta2 = beta[1]
        self.eps = epsilon
        self.decay = decay
        self.clip_norm = clip_norm
        self.m = [jnp.zeros_like(p.tensor) for p in param]
        self.v = [jnp.zeros_like(p.tensor) for p in param]

    def step(self):
        t = self.iterator
        new_m = []
        new_v = []

        for i, param in enumerate(self.params):
            g = param.grad
            if g is None:
                new_m.append(self.m[i])
                new_v.append(self.v[i])
                continue

            if self.clip_norm > 0.0:
                norm = jnp.linalg.norm(g)
                if norm > self.clip_norm :
                    scale = self.clip_norm / norm 
                    g = g * scale

            m = self.beta1 * self.m[i] + (1 - self.beta1) * g
            v = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            m_hat = m / (1 - self.beta1 ** (t + 1))
            v_hat = v / (1 - self.beta2 ** (t + 1))

            update = self.lr * m_hat / (jnp.sqrt(v_hat) + self.eps)

            param.tensor = param.tensor - update - self.lr * self.decay * param.tensor

            new_m.append(m)
            new_v.append(v)

        self.m = new_m
        self.v = new_v
        self.iterator += 1


class Adamax (Optimizer) :
    def __init__(self,param : list,lr = 1e-3,beta=(0.9,0.999),epsilon=1e-7,clip_norm=0.0) :
        super().__init__()
        for p in param :
            if not isinstance(p,Parameter) :
              raise TypeError("All elements in params must be Parameter objects")
        self.eps = epsilon
        self.params = param 
        self.beta1 = beta[0]
        self.beta2 = beta[0]

        self.lr = lr 
        self.clip_norm = clip_norm

        self.m = [jnp.zeros_like(p.tensor) for p in param]
        self.mn = [jnp.zeros_like(p.tensor) for p in param]
    
    def step(self):
        t = self.iterator
        new_m = []
        new_mn = []

        for i,param in enumerate(self.params) :
            g = param.grad 
            if g is None :
                new_m.append(self.m[i])
                new_mn.append(self.mn[i])
                continue

            if self.clip_norm > 0.0:
                norm = jnp.linalg.norm(g)
                if norm > self.clip_norm:
                    scale = self.clip_norm / norm 
                    g = g * scale

            
            m = self.beta1 * self.m[i] + (1 - self.beta1) * g 
            new_m.append(m)
            mn = jnp.maximum((self.beta2 * self.mn[i]),jnp.abs(g))
            new_mn.append(mn)
            m = m / (1 - self.beta2**(t+1))

            update = self.lr / (mn + self.eps) * m 
            param.tensor = param.tensor - update 
        
        self.m = new_m
        self.mn = new_mn
        self.iterator +=1 

class RMSProp (Optimizer) :
    def __init__ (self,param : list,lr=1e-3,beta=0.999,epsilon=1e-6,clip_norm=0.0) :
        super().__init__()
        for p in param :
            if not isinstance(p,Parameter) :
                raise TypeError("All elements in params must be Parameter objects")
        
        self.params = param 
        self.lr = lr 
        self.beta = beta 
        self.eps = epsilon
        self.rms = [jnp.zeros_like(p.tensor) for p in param]
        self.clip_norm = clip_norm
    
    def step(self) :
        t = self.iterator 
        new_rms  = []

        for i,param in enumerate(self.params) :
            g = param.grad 
            if g is None :
                new_rms.append(self.rms[i])
                continue
            if self.clip_norm > 0.0 :
                norm = jnp.linalg(g)
                if norm > self.clip_norm: 
                    g = g * (self.clip_norm / norm)
            
            rms = self.beta * self.rms[i] + (1 - self.beta) * (g**2)
            new_rms.append(rms)
            rms = rms / (1 - self.beta**(t+1))
            update = self.lr / (jnp.sqrt(rms ) +self.eps) * g 
            param.tensor = param.tensor - update

class Lion (Optimizer) :
    def __init__ (self,param : list,lr = 1e-3,beta=0.9,clip_norm=0.0) :
        super().__init__()
        for p in param :
            if not isinstance(p,Parameter) :
                raise TypeError("All elements in params must be Parameter objects")
        
        self.params = param 
        self.beta = beta 
        self.lr = lr 
        self.clip_norm = clip_norm
        self.m = [jnp.zeros_like(p.tensor) for p in param]
    
    def step(self) :
       
        new_m = []

        for i,param in enumerate(self.params) :

            g =  param.grad
            if g is None :
                new_m.append(self.m[i]) 
                continue
            if self.clip_norm > 0.0 :
                
                norm = jnp.linalg.norm(g)
                if norm  > self.clip_norm:
                    g = g * (self.clip_norm / norm)
                
            m = self.beta * self.m[i] + (1 - self.beta) * g 
            new_m.append(m)
            update = self.lr * jnp.sign(m)
            param.tensor = param.tensor - update 
        
        self.m = new_m

class AdaBelief (Optimizer) :
    def __init__ (self,param : list, lr : float = 1e-3,
                  beta=(0.9,0.999),epsilon=1e-6,clip_norm:float=0.0) :
        for p in param :
            if not isinstance(p,Parameter) :
                raise TypeError("All elements in params must be Parameter objects")
        super().__init__()
        self.lr = lr 
        self.model_param = param 
        self.m = [jnp.zeros_like(p.tensor) for p in param]
        self.s = [jnp.zeros_like(p.tensor) for p in param]
        self.beta1 = beta[0]
        self.beta2 = beta[1]
        self.eps = epsilon
        self.clip_norm = clip_norm
    
    def step(self):
        new_m = []
        new_s = []
        for i,param in enumerate(self.model_param) :
            g = param.grad
            if g is None :
                new_m.append(self.m[i])
                new_s.append(self.s[i])
                continue
            if self.clip_norm > 0.0 :
                norm = jnp.linalg.norm(g)
                if norm > self.clip_norm :
                    g = g * (self.clip_norm / norm)
            m = self.beta1 * self.m[i] + (1 - self.beta1 ) * g 
            s = self.beta2 * self.s[i] + (1 - self.beta2) * ((g - m) * (g-m))
            new_m.append(m)
            new_s.append(s)
            m = m / (1 - (self.beta1 ** (self.iterator + 1)))  
            s = s / (1 - (self.beta2 ** (self.iterator + 1)))
            update = self.lr * (m / jnp.sqrt(s + self.eps))
            param.tensor = param.tensor - update 
        self.m = new_m
        self.s = new_s
        self.iterator +=1 


class SGD (Optimizer):
    def __init__(self,param : list,lr=1e-3,clip_norm=0.0) :
        super().__init__()
        for p in param :
            if not isinstance(p,Parameter) :
                raise TypeError("All elements in params must be Parameter objects")
        self.params = param 
        self.lr = lr 
        self.clip_norm = clip_norm
    
    def step (self) :
        for param in self.params:
            g = param.grad 
            if g is None :
                continue
            if self.clip_norm > 0.0 :
                norm = jnp.linalg.norm(g)
                if norm > self.clip_norm :
                    g = g * (self.clip_norm / norm)
            update = self.lr * g
            param.tensor = param.tensor - update

class Momentum (Optimizer) :
    def __init__(self,param : list,lr=1e-3,momentum=0.9,clip_norm=0.0) :
        super().__init__()
        for p in param :
            if not isinstance(p,Parameter) :
                raise TypeError("All elements in params must be Parameter objects")
        self.params = param 
        self.lr = lr 
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.v = [jnp.zeros_like(p.tensor) for p in param]
    
    def step (self) :
        new_v = []
        for i,param in enumerate(self.params) :
            g = param.grad 
            if g is None :
                new_v.append(self.v[i])
                continue
            if self.clip_norm > 0.0 :
                norm = jnp.linalg.norm(g)
                if norm > self.clip_norm :
                    g = g * (self.clip_norm / norm)
            v = self.momentum * self.v[i] - self.lr * g 
            param.tensor = param.tensor + v 
            new_v.append(v)
        self.v = new_v

class Adagrad (Optimizer) :
    def __init__(self,param : list,lr=1e-2,epsilon=1e-7,clip_norm=0.0) :
        super().__init__()
        for p in param :
            if not isinstance(p,Parameter) :
                raise TypeError("All elements in params must be Parameter objects")
        self.params = param 
        self.lr = lr 
        self.eps = epsilon
        self.clip_norm = clip_norm
        self.accumulated_grad = [jnp.zeros_like(p.tensor) for p in param]
    
    def step (self) :
        new_accumulated_grad = []
        for i,param in enumerate(self.params) :
            g = param.grad 
            if g is None :
                new_accumulated_grad.append(self.accumulated_grad[i])
                continue
            if self.clip_norm > 0.0 :
                norm = jnp.linalg.norm(g)
                if norm > self.clip_norm :
                    g = g * (self.clip_norm / norm)
            accumulated_grad = self.accumulated_grad[i] + g**2
            new_accumulated_grad.append(accumulated_grad)
            update = self.lr / (jnp.sqrt(accumulated_grad) + self.eps) * g 
            param.tensor = param.tensor - update
        self.accumulated_grad = new_accumulated_grad

class NAdam (Optimizer) :
    def __init__ (self,param : list, lr = 1e-3, beta=(0.9,0.999),
                  epsilon = 1e-7, clip_norm=0.0) :
        super().__init__()
        for p in param :
            if not isinstance(p,Parameter) :
                raise TypeError("All elements in params must be Parameter objects")
        
        self.params = param 
        self.lr = lr
        self.beta1 = beta[0]
        self.beta2 = beta[1]
        self.eps = epsilon
        self.clip_norm = clip_norm
        self.m = [jnp.zeros_like(p.tensor) for p in param]
        self.v = [jnp.zeros_like(p.tensor) for p in param]
    
    def step(self) :
        nm = []
        nv = []
        for i,param in enumerate(self.params) :
            g = param.grad 
            if g is None :
                nm.append(self.m[i])
                nv.append(self.v[i])
                continue
            if self.clip_norm > 0.0 :
                norm = jnp.linalg.norm(g)
                if norm > self.clip_norm :
                    g = g * (self.clip_norm / norm)
            m = self.beta1 * self.m[i] + (1 - self.beta1) * g 
            v = self.beta2 * self.v[i] + (1 - self.beta2) * (g**2)
            m_hat = m / (1 - self.beta1**(self.iterator + 1))
            v_hat = v / (1 - self.beta2**(self.iterator + 1))
            m_n = (self.beta1 * m_hat) + ((1 - self.beta1) * g / (1 - self.beta1**(self.iterator + 1)))
            update = self.lr / (jnp.sqrt(v_hat) + self.eps) * m_n
            param.tensor = param.tensor - update
            nm.append(m)
            nv.append(v)
        self.m = nm
        self.v = nv
        self.iterator +=1
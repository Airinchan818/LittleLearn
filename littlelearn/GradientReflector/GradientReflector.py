import littlelearn as ll 
import weakref
import jax.numpy as jnp
import jax
import gc

class Node:
    def __init__(self, data, parents=(), backward=None):
        self.data = data
        self.grad = jnp.zeros_like(data)

        if parents is None:
            self.parents = ()
        else:
            
            self.parents = tuple(weakref.ref(p) for p in parents if p is not None )


        self.backward = backward if backward is not None else self._noop

    def _noop(self):
        pass

    def backwardpass(self):
        topo = []
        visited = set()
        def build(v):
            if v is None:
                return
            if id(v) in visited:
                return
            visited.add(id(v))

            for p_ref in v.parents:
                p = p_ref()
                if p is not None:
                    build(p)

            topo.append(v)

        build(self)


        self.grad = jnp.ones_like(self.data)


        for node in reversed(topo):
            if node is not None :
         
                node.backward()
          
    def reset_grad (self) :
        topo = list()
        visited = set()
        def build(v):
            if v is None:
                return
            if id(v) in visited:
                return
            visited.add(id(v))

            for p_ref in v.parents:
                p = p_ref()
                if p is not None:
                    build(p)

            topo.append(v)

        build(self)

        for node in topo :
            node.grad = jnp.zeros_like(node.data)
            node.parents = ()
            node.backward = node._noop

        gc.collect()

def adjust_shape(b, grad_shape):
    while b.ndim > len(grad_shape):
        b = jnp.sum(b, axis=0)

    while b.ndim < len(grad_shape):
        b = jnp.expand_dims(b, axis=0)

    for i in range(len(grad_shape)):
        if grad_shape[i] == 1 and b.shape[i] != 1:
            b = jnp.sum(b, axis=i, keepdims=True)

    return b
def align_or_reduce_grad(grad, tensor):

    if tensor.is_param:

        g = grad
        target_shape = tensor.tensor.shape


        while g.ndim > len(target_shape):
            g = g.sum(axis=0)

        for i, (gs, ts) in enumerate(zip(g.shape, target_shape)):
            if gs != ts:
                g = g.sum(axis=i, keepdims=True)

        return g.reshape(target_shape)
    else:

        return adjust_shape(grad, tensor.shape)


def im2col(x, KH, KW, stride, pad):
    N, C, H, W = x.shape
    H_out = (H + 2 * pad - KH) // stride + 1
    W_out = (W + 2 * pad - KW) // stride + 1

    x_padded = jnp.pad(
        x,
        ((0,0), (0,0), (pad,pad), (pad,pad))
    )

    i0 = jnp.repeat(jnp.arange(KH), KW)
    i0 = jnp.tile(i0, C)

    j0 = jnp.tile(jnp.arange(KW), KH)
    j0 = jnp.tile(j0, C)

    d = jnp.repeat(jnp.arange(C), KH * KW)

    i1 = stride * jnp.repeat(jnp.arange(H_out), W_out)
    j1 = stride * jnp.tile(jnp.arange(W_out), H_out)

    i = i0[:, None] + i1[None, :]
    j = j0[:, None] + j1[None, :]

    cols = x_padded[:, d[:, None], i[None, :, :], j[None, :, :]]
    cols = cols.reshape(N, C * KH * KW, H_out * W_out)

    return cols,x_padded


def col2im(cols, x_shape, KH, KW, stride, pad):
    N, C, H, W = x_shape
    H_out = (H + 2*pad - KH) // stride + 1
    W_out = (W + 2*pad - KW) // stride + 1

    dx_padded = jnp.zeros((N, C, H + 2*pad, W + 2*pad))

    i0 = jnp.repeat(jnp.arange(KH), KW)
    i0 = jnp.tile(i0, C)

    j0 = jnp.tile(jnp.arange(KW), KH)
    j0 = jnp.tile(j0, C)

    d  = jnp.repeat(jnp.arange(C), KH * KW)

    i1 = stride * jnp.repeat(jnp.arange(H_out), W_out)
    j1 = stride * jnp.tile(jnp.arange(W_out), H_out)

    i = i0[:, None] + i1[None, :]
    j = j0[:, None] + j1[None, :]

    cols_reshaped = cols.reshape(N, C*KH*KW, H_out*W_out)

    dx_padded = dx_padded.at[
        :, 
        d[:, None],
        i[None, :, :],
        j[None, :, :]
    ].add(cols_reshaped)

    if pad > 0:
        return dx_padded[:, :, pad:-pad, pad:-pad]

    return dx_padded

class Conv2DBackward:
    def __init__(self, x, w, b, stride, pad, cols, x_padded, out_shape, node):
        self.x = x
        self.w = w
        self.b = b if b is not None else None
        self.node = weakref.ref(node)


        self.stride = stride
        self.pad = pad
        self.cols = cols
        self.x_padded = x_padded
        self.out_shape = out_shape

    def __call__(self):
        node = self.node()
        x = self.x
        w = self.w
        b = self.b if self.b is not None else None

        if node is None:
            return
        
        dOut = node.grad  
        N, Cout, Hout, Wout = dOut.shape
        KH, KW = w.shape[2], w.shape[3]


        if b is not None and b.requires_grad:
            db = jnp.sum(dOut, axis=(0, 2, 3))
            b.node.grad = b.node.grad + db


        L = Hout * Wout
        dOut_flat = dOut.reshape(N, Cout, L)


        if w.requires_grad:
            dW = jnp.zeros_like(w.tensor)
            for n in range(N):
                dW += jnp.matmul(dOut_flat[n], self.cols[n].T).reshape(w.shape)
            w.node.grad = w.node.grad + dW

   
        if x.requires_grad:
            w_col = w.tensor.reshape(Cout, -1)
            dCols = jnp.zeros_like(self.cols)
            for n in range(N):
                dCols = dCols.at[n].set(jnp.matmul(w_col.T, dOut_flat[n]))

            dx = col2im(
                dCols,
                self.x_padded.shape,
                KH, KW,
                self.stride,
                self.pad
            )
            x.node.grad = x.node.grad + dx


class AddBackward:
    def __init__(self, node, a, b):
        self.node = weakref.ref(node)
        self.a = a
        self.b = b

    def __call__(self):
        node = self.node()
        if node is None:
            return

        grad_out = node.grad

        if self.a.requires_grad:
            a_grad = align_or_reduce_grad(grad_out, self.a)
            self.a.node.grad = (
                self.a.node.grad + a_grad
                if self.a.node.grad is not None
                else a_grad
            )

        if self.b.requires_grad:
            b_grad = align_or_reduce_grad(grad_out, self.b)
            self.b.node.grad = (
                self.b.node.grad + b_grad
                if self.b.node.grad is not None
                else b_grad
            )

class SubBackward:
    def __init__(self, node, a, b):
        self.node = weakref.ref(node)
        self.a = a
        self.b = b

    def __call__(self):
        node = self.node()
        if node is None:
            return

        grad_out = node.grad

        if self.a.requires_grad:
            a_grad = align_or_reduce_grad(grad_out, self.a)
            self.a.node.grad = (
                self.a.node.grad + a_grad
                if self.a.node.grad is not None
                else a_grad
            )

        if self.b.requires_grad:
            b_grad = align_or_reduce_grad(grad_out, self.b)
            self.b.node.grad = (
                self.b.node.grad - b_grad
                if self.b.node.grad is not None
                else -b_grad
            )


class PowBackward:
    def __init__(self, node, a, factor):
        self.node = weakref.ref(node)
        self.a = a
        self.factor = factor

    def __call__(self):
        node = self.node()
        if node is None or not self.a.requires_grad:
            return

        grad = node.grad
        base = self.a.tensor

        grad_a = self.factor * jnp.power(base, self.factor - 1) * grad
        self.a.node.grad = self.a.node.grad + align_or_reduce_grad(grad_a, self.a)
class MulBackward:
    def __init__(self, node, a, b):
        self.node = weakref.ref(node)
        self.a = a
        self.b = b

    def __call__(self):
        node = self.node()
        if node is None:
            return

        grad = node.grad

        if self.a.requires_grad:
            grad_a = grad * self.b.tensor
            self.a.node.grad = self.a.node.grad + align_or_reduce_grad(grad_a, self.a)

        if self.b.requires_grad:
            grad_b = grad * self.a.tensor
            self.b.node.grad = self.b.node.grad + align_or_reduce_grad(grad_b, self.b)


class DivBackward:
    def __init__(self, node, a, b):
        self.node =weakref.ref(node)
        self.a = a
        self.b = b

    def __call__(self):
        node = self.node()
        a = self.a
        b = self.b
        if node is None: return

        if a.requires_grad:
            a_grad = adjust_shape(node.grad, a.shape)
            node.grad = node.grad + (jnp.ones_like(a.tensor) / b.tensor) * a_grad

        if self.b.requires_grad:
            b_grad = adjust_shape(node.grad, self.b.shape)
            self.b.node.grad = self.b.node.grad + (-self.a.tensor / (self.b.tensor**2)) * b_grad

def conv2d_forward(x, w, b, stride, pad):
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = w.shape
    
    cols, x_padded = im2col(x, KH, KW, stride, pad)
    

    w_col = w.reshape(Cout, Cin * KH * KW)

    out = jnp.matmul(w_col[None], cols)  
    out = out.reshape(N, Cout, -1)      
    
    H_out = (H + 2*pad - KH) // stride + 1
    W_out = (W + 2*pad - KW) // stride + 1
    out = out.reshape(N, Cout, H_out, W_out)
    if b is not None:
        out = out + b.reshape(1, Cout, 1, 1)
    
    return out, cols, x_padded


class expbackward:
    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return

        if a.requires_grad:
            a_grad = node.grad
            a.node.grad = a.node.grad + (jnp.exp(a.tensor) * a_grad)

class TanhBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return

        if a.requires_grad:
            a_grad = node.grad
            a.node.grad = a.node.grad + (1 - jnp.tanh(a.tensor)**2) * a_grad


class LogBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a()
        if node is None or a is None:
            return

        if a.requires_grad:
            a_grad = node.grad
            a.node.grad = a.node.grad + (1 / a.tensor) * a_grad


class MatMulBackward:
    __slots__ = ("node", "a", "b", "transpose_a", "transpose_b")

    def __init__(self, node, a, b, transpose_a=False, transpose_b=False):
        self.node = weakref.ref(node)
        self.a = a
        self.b = b
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def __call__(self):
        node = self.node()
        a = self.a
        b = self.b
        if node is None or a is None or b is None:
            return

        grad = node.grad

        A = a.tensor
        B = b.tensor

        if self.transpose_a:
            A = jnp.swapaxes(A, -2, -1)
        if self.transpose_b:
            B = jnp.swapaxes(B, -2, -1)

        if a.requires_grad:
            gA = jnp.matmul(grad, jnp.swapaxes(B, -2, -1))
            gA = align_or_reduce_grad(gA,tensor=a)
            if self.transpose_a:
                a.node.grad = a.node.grad + gA.swapaxes(-2, -1)
            else:
                a.node.grad = a.node.grad + gA

        if b.requires_grad:
            gB = jnp.matmul(jnp.swapaxes(A, -2, -1), grad)
            gB = align_or_reduce_grad(gB,tensor=b)
            if self.transpose_b:
                b.node.grad = b.node.grad + gB.swapaxes(-2, -1)
            else:
                b.node.grad = b.node.grad + gB


class SumBackward:
    __slots__ = ("node", "a", "axis", "keepdims")

    def __init__(self, node, a, axis=None, keepdims=False):
        self.node = weakref.ref(node)
        self.a = a
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return

        if a.requires_grad:
            grad = node.grad

            a_grad = jnp.broadcast_to(
                grad if self.keepdims else jnp.expand_dims(grad, axis=self.axis),
                a.shape
            )

            a.node.grad = a.node.grad + a_grad


class ReshapeBackward:
    __slots__ = ("node_ref", "a_ref", "a_shape")

    def __init__(self, node, a):
        self.node_ref = weakref.ref(node)
        self.a_ref = a
        self.a_shape = a.shape

    def __call__(self):
        node = self.node_ref()
        a = self.a_ref
        if node is None or a is None: return
        if a.requires_grad:
            a.node.grad = a.node.grad + jnp.reshape(node.grad, self.a_shape)



class TransposeBackward:
    __slots__ = ("node_ref", "a_ref", "axes")

    def __init__(self, node, a, axes=None):
        self.node_ref = weakref.ref(node)
        self.a_ref = a
        self.axes = axes

    def __call__(self):
        node = self.node_ref()
        a = self.a_ref
        if node is None or a is None: return

        if a.requires_grad:
            a_grad = node.grad
            if self.axes is None:
                reversed_axes = None
            else:
                reversed_axes = [0] * len(self.axes)
                for i, ax in enumerate(self.axes):
                    reversed_axes[ax] = i

            a.node.grad = a.node.grad + jnp.transpose(a_grad, axes=reversed_axes)



class FlattenBackward:
    __slots__ = ("node_ref", "a_ref", "a_shape")

    def __init__(self, node, a):
        self.node_ref = weakref.ref(node)
        self.a_ref = a
        self.a_shape = a.shape

    def __call__(self):
        node = self.node_ref()
        a = self.a_ref
        if node is None or a is None: return

        if a.requires_grad:
            a.node.grad = a.node.grad + jnp.reshape(node.grad, self.a_shape)



class GetItemBackward:
    __slots__ = ("node_ref", "a_ref", "idx", "a_shape")

    def __init__(self, node, a, idx):
        self.node_ref = weakref.ref(node)
        self.a_ref = a
        self.idx = idx
        self.a_shape = a.shape

    def __call__(self):
        node = self.node_ref()
        a = self.a_ref
        if a.requires_grad:
            grad_full = jnp.zeros_like(a.tensor)
            grad_full = grad_full.at[self.idx].set(node.grad)
            a.node.grad = a.node.grad + grad_full



class MaxBackward:
    __slots__ = ("node_ref", "a_ref", "axis", "keepdims", "output")

    def __init__(self, node, a, axis=None, keepdims=False, output=None):
        self.node_ref = weakref.ref(node)
        self.a_ref = a
        self.axis = axis
        self.keepdims = keepdims
        self.output = output

    def __call__(self):
        node = self.node_ref()
        a = self.a_ref
        if node is None or a is None: return
        
        if a.requires_grad:
            a_grad = adjust_shape(node.grad, a.shape)

            if self.keepdims:
                max_mask = a.tensor == jnp.expand_dims(self.output, axis=self.axis)
            else:
                max_mask = a.tensor == jnp.expand_dims(self.output, axis=self.axis).squeeze(axis=self.axis)

            grad_full = max_mask * a_grad
            a.node.grad = a.node.grad + grad_full



class MinBackward:
    __slots__ = ("node_ref", "a_ref", "axis", "keepdims", "output")

    def __init__(self, node, a, axis=None, keepdims=False, output=None):
        self.node_ref = weakref.ref(node)
        self.a_ref = a
        self.axis = axis
        self.keepdims = keepdims
        self.output = output

    def __call__(self):
        node = self.node_ref()
        a = self.a_ref
        if node is None or a is None: return
        
        if a.requires_grad:
            a_grad = adjust_shape(node.grad, a.shape)

            if self.keepdims:
                min_mask = a.tensor == jnp.expand_dims(self.output, axis=self.axis)
            else:
                min_mask = a.tensor == jnp.expand_dims(self.output, axis=self.axis).squeeze(axis=self.axis)

            grad_full = min_mask * a_grad
            a.node.grad = a.node.grad + grad_full



class WhereBackward:
    __slots__ = ("node_ref", "cond_ref", "a_ref", "b_ref")

    def __init__(self, node, condition, a, b):
        self.node_ref = weakref.ref(node)
        self.cond_ref = condition
        self.a_ref = a
        self.b_ref = b

    def __call__(self):
        node = self.node_ref()
        cond = self.cond_ref
        a = self.a_ref
        b = self.b_ref
        if node is None: return

        if a and a.requires_grad:
            a_grad = node.grad
            grad_full = cond.tensor * a_grad
            a.node.grad = a.node.grad + grad_full

        if b and b.requires_grad:
            b_grad = node.grad
            grad_full = (1 - cond.tensor) * b_grad
            b.node.grad = b.node.grad + grad_full


class AbsBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            sign = jnp.sign(a.tensor)
            a.node.grad = a.node.grad + (sign * a_grad)


class ClipBackward:
    __slots__ = ("node", "a", "min_value", "max_value")

    def __init__(self, node, a, min_value, max_value):
        self.node = weakref.ref(node)
        self.a = a
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            mask = ((a.tensor >= self.min_value) & (a.tensor <= self.max_value)).astype(a.tensor.dtype)
            a.node.grad = a.node.grad + (mask * a_grad)


class ExpandDimsBackward:
    __slots__ = ("node", "a", "a_shape")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a
        self.a_shape = a.shape

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            a.node.grad = a.node.grad + jnp.reshape(a_grad, self.a_shape)


class SqueezeBackward:
    __slots__ = ("node", "a", "a_shape")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a
        self.a_shape = a.shape

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            a.node.grad = a.node.grad + jnp.reshape(a_grad, self.a_shape)


class UnsqueezeBackward:
    __slots__ = ("node", "a", "a_shape")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a
        self.a_shape = a.shape

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            a.node.grad = a.node.grad + jnp.reshape(a_grad, self.a_shape)


class GeluTanhBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            sqrt_2_over_pi = jnp.sqrt(2 / jnp.pi)
            tanh_arg = sqrt_2_over_pi * (a.tensor + 0.044715 * jnp.power(a.tensor, 3))
            tanh_out = jnp.tanh(tanh_arg)
            sech2 = 1 - jnp.power(tanh_out, 2)
            pdf = sqrt_2_over_pi * (1 + 3 * 0.044715 * jnp.power(a.tensor, 2)) * sech2
            cdf = 0.5 * (1.0 + tanh_out)
            grad = cdf + a.tensor * pdf
            a.node.grad = a.node.grad + grad * a_grad


class SoftplusBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            sigmoid = 1 / (1 + jnp.exp(-a.tensor))
            a.node.grad = a.node.grad + (sigmoid * a_grad)


class CeluBackward:
    __slots__ = ("node", "a", "alpha")

    def __init__(self, node, a, alpha=1.0):
        self.node = weakref.ref(node)
        self.a = a
        self.alpha = alpha

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = jnp.where(a.tensor > 0, 1, jnp.exp(a.tensor / self.alpha))
            a.node.grad = a.node.grad + (grad * a_grad)


class SeluBackward:
    __slots__ = ("node", "a", "lambda_", "alpha")

    def __init__(self, node, a, lambda_=1.0507, alpha=1.673):
        self.node = weakref.ref(node)
        self.a = a
        self.lambda_ = lambda_
        self.alpha = alpha

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = jnp.where(
                a.tensor > 0,
                self.lambda_,
                self.lambda_ * self.alpha * jnp.exp(a.tensor)
            )
            a.node.grad = a.node.grad + (grad * a_grad)


class SoftsignBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            denom = (1 + jnp.abs(a.tensor)) ** 2
            grad = 1 / denom
            a.node.grad = a.node.grad + (grad * a_grad)


class LogSoftmaxBackward:
    __slots__ = ("node", "a", "axis", "output", "keepdims")

    def __init__(self, node, a, axis=None, output=None, keepdims=False):
        self.node = weakref.ref(node)
        self.a = a
        self.axis = axis
        self.keepdims = keepdims
        self.output = output

    def __call__(self):
        node = self.node()
        a = self.a
        output = self.output
        if node is None or a is None or output is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            softmax = jnp.exp(output)
            sum_grad = jnp.sum(a_grad, axis=self.axis, keepdims=self.keepdims)
            grad = a_grad - softmax * sum_grad
            a.node.grad = a.node.grad + grad


class SoftmaxBackward:
    __slots__ = ("node", "x", "axis", "with_crossentropy", "output", "keepdims")

    def __init__(self, node, x, axis=None, with_crossentropy=False, output=None, keepdims=False):
        self.node = weakref.ref(node)
        self.x =x
        self.axis = axis
        self.with_crossentropy = with_crossentropy
        self.output = output
        self.keepdims = keepdims

    def __call__(self):
        node = self.node()
        x = self.x
        output = self.output
        if node is None or x is None:
            return

        x_grad = node.grad
        if x.requires_grad:
            if self.with_crossentropy:
                x.node.grad = x.node.grad + x_grad
            else:
                if output is None:
                    return
                grad = output * (
                    x_grad - jnp.sum(output * x_grad, axis=self.axis, keepdims=self.keepdims)
                )
                x.node.grad = x.node.grad + grad


class PadBackward:
    __slots__ = ("node", "a", "pad_width")

    def __init__(self, node, a, pad_width):
        self.node = weakref.ref(node)
        self.a = a
        self.pad_width = pad_width

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            slices = tuple(
                slice(pad[0], a.tensor.shape[i] + pad[0])
                for i, pad in enumerate(self.pad_width)
            )
            grad_full = jnp.zeros_like(node.data)
            grad_full = grad_full.at[slices].set(a_grad)
            a.node.grad = a.node.grad + grad_full[slices]


class BroadcastToBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            reduce_axes = tuple(range(a_grad.ndim - a.tensor.ndim))
            a.node.grad = a.node.grad + jnp.sum(a_grad, axis=reduce_axes)


class EluBackward:
    __slots__ = ("node", "a", "alpha")

    def __init__(self, node, a, alpha=1.0):
        self.node = weakref.ref(node)
        self.a = a
        self.alpha = alpha

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = jnp.where(a.tensor > 0, 1, self.alpha * jnp.exp(a.tensor))
            a.node.grad = a.node.grad + (grad * a_grad)


class ErfBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = (2 / jnp.sqrt(jnp.pi)) * jnp.exp(-jnp.power(a.tensor, 2))
            a.node.grad = a.node.grad + (grad * a_grad)


class SinhBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = jnp.cosh(a.tensor)
            a.node.grad = a.node.grad + (grad * a_grad)


class CoshBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = jnp.sinh(a.tensor)
            a.node.grad = a.node.grad + (grad * a_grad)


class TanBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = 1 / jnp.cos(a.tensor) ** 2
            a.node.grad = a.node.grad + (grad * a_grad)


class SumProductBackward:
    __slots__ = ("node", "a", "b")

    def __init__(self, node, a, b):
        self.node = weakref.ref(node)
        self.a = a
        self.b = b

    def __call__(self):
        node = self.node()
        a = self.a
        b = self.b
        if node is None or a is None or b is None:
            return

        if a.requires_grad:
            a_grad = node.grad
            a.node.grad = a.node.grad + (b.tensor * a_grad)

        if b.requires_grad:
            b_grad = node.grad
            b.node.grad = b.node.grad + (a.tensor * b_grad)


class Log10Backward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            a.node.grad = a.node.grad + (
                1 / (a.tensor * jnp.log(10)) * a_grad
            )


class SqrtBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None or not a.requires_grad:
            return

        grad_out = node.grad
        grad_out = align_or_reduce_grad(grad_out, a)

        grad = 0.5 / jnp.sqrt(a.tensor) * grad_out

        a.node.grad = (
            a.node.grad + grad
            if a.node.grad is not None
            else grad
        )


class Log2Backward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            a.node.grad = a.node.grad + (
                1 / (a.tensor * jnp.log(2)) * a_grad
            )


class FloorBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = jnp.zeros_like(a.tensor)
            a.node.grad = a.node.grad + (grad * a_grad)


class CeilBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = jnp.zeros_like(a.tensor)
            a.node.grad = a.node.grad + (grad * a_grad)


class RintBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = jnp.zeros_like(a.tensor)
            a.node.grad = a.node.grad + (grad * a_grad)


class ReciprocalBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            a.node.grad = a.node.grad + (-1 / (a.tensor ** 2) * a_grad)


class SignBackward:
    __slots__ = ("a", "node")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = jnp.zeros_like(a.tensor)
            a.node.grad = a.node.grad + (grad * a_grad)


class IdentityBackward:
    __slots__ = ("a", "node")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            a.node.grad = a.node.grad + (jnp.ones_like(a.tensor) * a_grad)


class Relu6Backward:
    __slots__ = ("a", "node")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = jnp.where((a.tensor > 0) & (a.tensor < 6), 1, 0)
            a.node.grad = a.node.grad + (grad * a_grad)


class HardShrinkBackward:
    __slots__ = ("node", "a", "lambda_")

    def __init__(self, node, a, lambda_=0.5):
        self.node = weakref.ref(node)
        self.a = a
        self.lambda_ = lambda_

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = jnp.where(jnp.abs(a.tensor) > self.lambda_, 1, 0)
            a.node.grad = a.node.grad + (grad * a_grad)


class HardTanhBackward:
    __slots__ = ("node", "a", "min_val", "max_val")

    def __init__(self, node, a, min_val=-1.0, max_val=1.0):
        self.node = weakref.ref(node)
        self.a = a
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = jnp.where(
                (a.tensor > self.min_val) & (a.tensor < self.max_val), 1, 0
            )
            a.node.grad = a.node.grad + (grad * a_grad)


class Log1pBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            a.node.grad = a.node.grad + (1 / (1 + a.tensor)) * a_grad


class HardSigmoidBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        a_grad = node.grad
        grad = jnp.where((a.tensor > -1) & (a.tensor < 1), 0.5, 0)
        a.node.grad = a.node.grad + (grad * a_grad)


class HardMaxBackward:
    __slots__ = ("node", "a", "axis", "keepdims")

    def __init__(self, node, a, axis=None, keepdims=False):
        self.node = weakref.ref(node)
        self.a = a
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = jnp.zeros_like(a.tensor)
            a.node.grad = a.node.grad + (grad * a_grad)


class LogSigmoidBackward:
    __slots__ = ("node", "a")

    def __init__(self, node, a):
        self.node = weakref.ref(node)
        self.a = a

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            sigmoid = 1 / (1 + jnp.exp(-a.tensor))
            a.node.grad = a.node.grad + (sigmoid * a_grad)


class ConcatBackward:
    def __init__(self, node, axis=0):
        self.node = weakref.ref(node)
        self.axis = axis

    def __call__(self):
        node = self.node()
        if node is None:
            return

        grad = node.grad

        shapes = [parent_ref().data.shape for parent_ref in node.parents if parent_ref() is not None]
        sizes = [s[self.axis] for s in shapes]

        indices = jnp.cumsum(jnp.array(sizes))[:-1].tolist()
        splits = jnp.split(grad, indices, axis=self.axis)

        for parent_ref, g in zip(node.parents, splits):
            parent = parent_ref()
            if parent is None:
                continue

            g = g.reshape(parent.data.shape)

            if parent.grad is None:
                parent.grad = g
            else:
                parent.grad = parent.grad + g


class SigmoidBackward:
    __slots__ = ("node", "a", "output")

    def __init__(self, node, a, output):
        self.node = weakref.ref(node)
        self.a = a
        self.output = output

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = self.output * (1 - self.output)
            a.node.grad = a.node.grad + (a_grad * grad)


class LogSumExpBackward:
    __slots__ = ("node", "a", "axis", "keepdims", "shifted_a", "sum_exp")

    def __init__(self, node, a, axis, keepdims, shifted_a, sum_exp):
        self.node = weakref.ref(node)
        self.a = a
        self.axis = axis
        self.keepdims = keepdims
        self.shifted_a = shifted_a
        self.sum_exp = sum_exp

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            softmax = jnp.exp(self.shifted_a) / self.sum_exp
            grad = softmax * adjust_shape(node.grad, softmax.shape)
            a.node.grad = a.node.grad + (grad * a_grad)


class SwishBackward:
    __slots__ = ("node", "a", "sigmoid", "beta")

    def __init__(self, node, a, sigmoid, beta):
        self.node = weakref.ref(node)
        self.a = a
        self.sigmoid = sigmoid
        self.beta = beta

    def __call__(self):
        node = self.node()
        a = self.a
        if node is None or a is None:
            return
        if a.requires_grad:
            a_grad = node.grad
            grad = self.sigmoid + a.tensor * self.sigmoid * (1 - self.sigmoid) * self.beta
            a.node.grad = a.node.grad + grad * a_grad


class LogAddExpBackward:
    __slots__ = ("node", "a", "b")

    def __init__(self, node, a, b):
        self.node = weakref.ref(node)
        self.a = a
        self.b = b

    def __call__(self):
        node = self.node()
        a = self.a
        b = self.b
        if node is None or a is None or b is None:
            return

        exp_a = jnp.exp(a.tensor)
        exp_b = jnp.exp(b.tensor)
        denom = exp_a + exp_b

        if a.requires_grad:
            a_grad = adjust_shape(node.grad, a.shape)
            a.node.grad = a.node.grad + (exp_a / denom) * a_grad

        if b.requires_grad:
            b_grad = adjust_shape(node.grad, b.shape)
            b.node.grad = b.node.grad + (exp_b / denom) * b_grad


class ReLUBackward:
    __slots__ = ("node", "x")

    def __init__(self, node, x):
        self.node = weakref.ref(node)
        self.x = x

    def __call__(self):
        node = self.node()
        x = self.x
        if node is None or x is None:
            return
        if x.requires_grad:
            grad = node.grad
            x.node.grad = x.node.grad + (jnp.where(x.tensor < 0, 0, 1) * grad)


class LeakyReLUBackward:
    __slots__ = ("node", "x", "alpha")

    def __init__(self, node, x, alpha):
        self.node = weakref.ref(node)
        self.x = x
        self.alpha = alpha

    def __call__(self):
        node = self.node()
        x = self.x
        if node is None or x is None:
            return
        if x.requires_grad:
            x_grad = node.grad
            grad_mask = jnp.where(x.tensor > 0, 1, self.alpha)
            x.node.grad = x.node.grad + grad_mask * x_grad


class MeanBackward:
    __slots__ = ("node", "x", "axis", "keepdims")

    def __init__(self, node, x, axis, keepdims):
        self.node = weakref.ref(node)
        self.x = x
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self):
        node = self.node()
        x = self.x
        if node is None or x is None or not x.requires_grad:
            return

        grad_out = node.grad

        if self.axis is None:
            N = x.tensor.size
        elif isinstance(self.axis, tuple):
            N = int(jnp.prod(jnp.array([x.tensor.shape[a] for a in self.axis])))
        else:
            N = x.tensor.shape[self.axis]

        if self.axis is not None and not self.keepdims:
            if isinstance(self.axis, tuple):
                for ax in sorted(self.axis):
                    grad_out = jnp.expand_dims(grad_out, axis=ax)
            else:
                grad_out = jnp.expand_dims(grad_out, axis=self.axis)

        grad_input = jnp.broadcast_to(grad_out, x.tensor.shape) / N

   
        grad_input = align_or_reduce_grad(grad_input, x)

        x.node.grad = (
            x.node.grad + grad_input
            if x.node.grad is not None
            else grad_input
        )


class VarBackward:
    __slots__ = ("node", "x", "mean", "axis", "keepdims")

    def __init__(self, node, x, mean, axis, keepdims):
        self.node = weakref.ref(node)
        self.x = x
        self.mean = mean
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self):
        node = self.node()
        x = self.x
        if node is None or x is None or not x.requires_grad:
            return

        if self.axis is None:
            N = x.tensor.size
            mean = self.mean
        else:
            N = x.tensor.shape[self.axis]
            mean = jnp.expand_dims(self.mean, axis=self.axis)

        grad_out = node.grad
        grad_out = align_or_reduce_grad(grad_out, x)

        grad = (2.0 / N) * (x.tensor - mean)
        x.node.grad = (
            x.node.grad + grad * grad_out
            if x.node.grad is not None
            else grad * grad_out
        )


class StdBackward:
    __slots__ = ("node", "x", "mean", "std", "axis", "keepdims")

    def __init__(self, node, x, mean, std, axis, keepdims):
        self.node = weakref.ref(node)
        self.x = x
        self.mean = mean
        self.std = std
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self):
        node = self.node()
        x = self.x
        if node is None or x is None or not x.requires_grad:
            return

        if self.axis is None:
            N = jnp.prod(jnp.array(x.shape))
            mean = self.mean
            std = self.std
        else:
            N = x.shape[self.axis]
            mean = jnp.expand_dims(self.mean, axis=self.axis)
            std = jnp.expand_dims(self.std, axis=self.axis)

        grad_out = align_or_reduce_grad(node.grad, x)
        grad = (x.tensor - mean) / (N * std)

        x.node.grad = (
            x.node.grad + grad * grad_out
            if x.node.grad is not None
            else grad * grad_out
        )




class MSELossBackward:
    __slots__ = ("node", "y_pred", "y_train")

    def __init__(self, node, y_pred, y_train):
        self.node = weakref.ref(node)
        self.y_pred = y_pred
        self.y_train = y_train

    def __call__(self):
        node = self.node()
        y_pred = self.y_pred
        if node is None or y_pred is None:
            return
        if y_pred.requires_grad:
            a_grad = node.grad
            grad = (2 / self.y_train.shape[0]) * (y_pred.tensor - self.y_train)
            y_pred.node.grad = y_pred.node.grad + grad * a_grad


class MAELossBackward:
    __slots__ = ("node", "y_pred", "y_train")

    def __init__(self, node, y_pred, y_train):
        self.node = weakref.ref(node)
        self.y_pred = y_pred
        self.y_train = y_train

    def __call__(self):
        node = self.node()
        y_pred = self.y_pred
        if node is None or y_pred is None:
            return
        if y_pred.requires_grad:
            a_grad = node.grad
            grad = (1 / self.y_train.shape[0]) * jnp.sign(y_pred.tensor - self.y_train)
            y_pred.node.grad = y_pred.node.grad + grad * a_grad


class HuberLossBackward:
    __slots__ = ("node", "y_pred", "y_train", "delta", "loss")

    def __init__(self, node, y_pred, y_train, delta, loss):
        self.node = weakref.ref(node)
        self.y_pred = y_pred
        self.y_train = y_train
        self.delta = delta
        self.loss = loss

    def __call__(self):
        node = self.node()
        y_pred = self.y_pred
        if node is None or y_pred is None:
            return
        if y_pred.requires_grad:
            a_grad = node.grad
            grad = jnp.where(
                jnp.abs(self.loss) <= self.delta,
                self.loss,
                self.delta * jnp.sign(self.loss)
            )
            y_pred.node.grad = y_pred.node.grad + grad * a_grad


class BCELossBackward:
    __slots__ = ("node", "y_pred", "y_train", "epsilon")

    def __init__(self, node, y_pred, y_train, epsilon):
        self.node = weakref.ref(node)
        self.y_pred = y_pred
        self.y_train = y_train
        self.epsilon = epsilon

    def __call__(self):
        node = self.node()
        y_pred = self.y_pred
        if node is None or y_pred is None:
            return
        if y_pred.requires_grad:
            a_grad = node.grad
            pred = jnp.clip(y_pred.tensor, self.epsilon, 1 - self.epsilon)
            grad = (1 / self.y_train.shape[0]) * (
                (pred - self.y_train) / (pred * (1 - pred))
            )
            y_pred.node.grad = y_pred.node.grad + grad * a_grad


class SparseCrossEntropyBackward:
    def __init__(self, node, logits, y_true):
        self.node = weakref.ref(node)
        self.logits = logits
        self.y_true = y_true

    def __call__(self):
        node = self.node()
        logits = self.logits
        if node is None or logits is None:
            return

        orig_shape = logits.shape
        num_classes = orig_shape[-1]

        flat_logits = logits.tensor.reshape(-1, num_classes)
        flat_labels = jnp.array(self.y_true).reshape(-1)
        N = flat_labels.shape[0]

        softmax_val = jax.nn.softmax(flat_logits, axis=-1)

        grad = softmax_val
        grad = grad.at[jnp.arange(N), flat_labels].add(-1.0)
        grad = grad / N

        grad = grad.reshape(orig_shape)
        logits.node.grad = logits.node.grad + grad


class CrossEntropyBackward:
    def __init__(self, node, logits, y_true):
        self.node = weakref.ref(node)
        self.logits = logits
        self.y_true = y_true

    def __call__(self):
        node = self.node()
        logits = self.logits
        if node is None or logits is None:
            return

        orig_shape = logits.shape
        num_classes = orig_shape[-1]

        flat_logits = logits.tensor.reshape(-1, num_classes)
        flat_y = self.y_true.reshape(-1, num_classes)
        N = flat_y.shape[0]

        softmax_val = jax.nn.softmax(flat_logits, axis=-1)
        grad = (softmax_val - flat_y) / N
        grad = grad.reshape(orig_shape)

        logits.node.grad = logits.node.grad + grad


class MaskedCrossEntropyBackward:
    def __init__(self, node, logits, y_true, mask):
        self.node = weakref.ref(node)
        self.logits = logits
        self.y_true = y_true
        self.mask = mask

    def __call__(self):
        node = self.node()
        logits = self.logits
        if node is None or logits is None:
            return

        logits = logits.tensor
        y = self.y_true
        mask = self.mask

        orig_shape = logits.shape
        num_classes = orig_shape[-1]

        flat_logits = logits.reshape(-1, num_classes)
        flat_y = y.reshape(-1, num_classes)
        flat_mask = mask.reshape(-1)

        softmax_val = jax.nn.softmax(flat_logits, axis=-1)

        grad = softmax_val - flat_y
        grad = grad * flat_mask[:, None]

        N = jnp.sum(flat_mask) + 1e-8
        grad = grad / N

        grad = grad.reshape(orig_shape)
        logits.node.grad = logits.node.grad + grad


class MaskedSparseCrossEntropyBackward:
    def __init__(self, node, logits, y_true, mask):
        self.node = weakref.ref(node)
        self.logits = logits
        self.y_true = y_true
        self.mask = mask

    def __call__(self):
        node = self.node()
        logits = self.logits
        if node is None or logits is None:
            return

        labels = jnp.array(self.y_true).reshape(-1)
        mask = self.mask.reshape(-1)

        orig_shape = logits.shape
        num_classes = orig_shape[-1]

        flat_logits = logits.reshape(-1, num_classes)

        softmax_val = jax.nn.softmax(flat_logits, axis=-1)

        grad = softmax_val
        grad = grad.at[jnp.arange(labels.shape[0]), labels].add(-1.0)
        grad = grad * mask[:, None]

        N = jnp.sum(mask) + 1e-8
        grad = grad / N

        grad = grad.reshape(orig_shape)
        logits.node.grad = logits.node.grad + grad


class MaxPool2DBackward:
    def __init__(self, node, x, kernel, stride, padding, argmax_mask):
        self.node = weakref.ref(node)
        self.x = x
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.argmax_mask = argmax_mask  

    def __call__(self):
        node = self.node()
        x_ref = self.x
        if node is None or x_ref is None:
            return

        x = x_ref.tensor
        N, H, W, C = x.shape
        Kh, Kw = self.kernel
        Sh, Sw = self.stride
        pad = self.padding

        grad_out = adjust_shape(node.grad, node.data.shape)
        Ho, Wo = grad_out.shape[1], grad_out.shape[2]

        grad_input = jnp.zeros_like(x)

        for i in range(Ho):
            for j in range(Wo):
                h_start = i * Sh - pad
                w_start = j * Sw - pad
                h_end = h_start + Kh
                w_end = w_start + Kw

                patch_mask = self.argmax_mask[:, i, j]
                grad_patch = grad_out[:, i, j, :]
                grad_patch = grad_patch[:, :, None, None] * patch_mask

                grad_input = grad_input.at[
                    :,
                    h_start:h_end,
                    w_start:w_end,
                    :
                ].add(jnp.moveaxis(grad_patch, [1, 2, 3], [3, 1, 2]))

        x_ref.node.grad = x_ref.node.grad + grad_input

class lstm_cell_backward:
    def __init__(self, node, i, f, g,
                 x_t, h_prev, c_prev,
                 Wi, Wf, Wg,
                 Ui, Uf, Ug):

        self.node = weakref.ref(node)
        self.i = i
        self.f = f
        self.g = g

        self.x_t = x_t
        self.h_prev = h_prev
        self.c_prev = c_prev

        self.Wi = Wi
        self.Wf = Wf
        self.Wg = Wg

        self.Ui = Ui
        self.Uf = Uf
        self.Ug = Ug

    def __call__(self):
        node = self.node()
        x_t = self.x_t
        h_prev = self.h_prev
        c_prev = self.c_prev
        Wi = self.Wi
        Wf = self.Wf
        Wg = self.Wg
        Ui = self.Ui
        Uf = self.Uf
        Ug = self.Ug

        if (node is None or x_t is None or h_prev is None or c_prev is None or
            Wi is None or Wf is None or Wg is None or
            Ui is None or Uf is None or Ug is None):
            return

        dc = node.grad

        df = dc * c_prev.tensor
        di = dc * self.g
        dg = dc * self.i
        dc_prev = dc * self.f

        df_raw = df * self.f * (1 - self.f)
        di_raw = di * self.i * (1 - self.i)
        dg_raw = dg * (1 - self.g * self.g)

        dx = (
            jnp.matmul(df_raw, Wf.tensor.T) +
            jnp.matmul(di_raw, Wi.tensor.T) +
            jnp.matmul(dg_raw, Wg.tensor.T)
        )

        dh = (
            jnp.matmul(df_raw, Uf.tensor.T) +
            jnp.matmul(di_raw, Ui.tensor.T) +
            jnp.matmul(dg_raw, Ug.tensor.T)
        )

        Wf.node.grad = Wf.node.grad + jnp.matmul(x_t.tensor.T, df_raw)
        Wi.node.grad = Wi.node.grad + jnp.matmul(x_t.tensor.T, di_raw)
        Wg.node.grad = Wg.node.grad + jnp.matmul(x_t.tensor.T, dg_raw)

        Uf.node.grad = Uf.node.grad + jnp.matmul(h_prev.tensor.T, df_raw)
        Ui.node.grad = Ui.node.grad + jnp.matmul(h_prev.tensor.T, di_raw)
        Ug.node.grad = Ug.node.grad + jnp.matmul(h_prev.tensor.T, dg_raw)

        x_t.node.grad = x_t.node.grad + dx
        h_prev.node.grad = h_prev.node.grad + dh
        c_prev.node.grad = c_prev.node.grad + dc_prev


class lstm_hidden_backward:
    def __init__(self, node, o, hc,
                 x_t, h_prev, cell_state,
                 w_output, wh_output):

        self.node = weakref.ref(node)
        self.o = o
        self.hc = hc

        self.x_t = x_t
        self.h_prev = h_prev
        self.cell_state = cell_state

        self.w_output = w_output
        self.wh_output = wh_output

    def __call__(self):
        node = self.node()
        x_t = self.x_t
        h_prev = self.h_prev
        cell_state = self.cell_state
        w_output = self.w_output
        wh_output = self.wh_output

        if (node is None or x_t is None or h_prev is None or
            cell_state is None or w_output is None or wh_output is None):
            return

        dh = node.grad

        do = dh * self.hc
        do_raw = do * self.o * (1.0 - self.o)
        dc = dh * self.o * (1.0 - self.hc * self.hc)

        w_output.node.grad = w_output.node.grad + jnp.matmul(x_t.tensor.T, do_raw)
        wh_output.node.grad = wh_output.node.grad + jnp.matmul(h_prev.tensor.T, do_raw)

        dx = jnp.matmul(do_raw, w_output.tensor.T)
        dh_prev = jnp.matmul(do_raw, wh_output.tensor.T)

        x_t.node.grad = x_t.node.grad + dx
        h_prev.node.grad = h_prev.node.grad + dh_prev
        cell_state.node.grad = cell_state.node.grad + dc

class SinBackward :
    def __init__(self,node,x) : 
        self.node = weakref.ref(node)
        self.x = x 
    
    def __call__ (self) :
        node = self.node()
        if self.x.requires_grad :
          
            self.x.node.grad = self.x.node.grad + (jnp.cos(self.x.tensor) * node.grad)

class CosBackward :

    def __init__(self,node,x) :
        self.node = weakref.ref(node)
        self.x = x 
        
    
    def __call__(self):
        node = self.node()
        if self.x.reuires_grad :
            self.x.node.grad = self.x.node.grad + (jnp.sin(self.x.tensor) * node.grad)


class GradientReflector:

    @staticmethod
    def add(a, b):
        output = a.tensor + b.tensor
        if not (a.requires_grad or b.requires_grad):
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)

        node = Node(output, parents=(a.node, b.node))
        node.backward = AddBackward(node, a, b)
        return ll.Tensor(data=output, requires_grad=True, _node=node,
                         dtype=a.dtype,device=a.device)


    @staticmethod
    def subtract(a, b):
        output = a.tensor - b.tensor
        if not (a.requires_grad or b.requires_grad):
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)

        node = Node(output, parents=(a.node, b.node))
        node.backward = SubBackward(node, a, b)
        return ll.Tensor(data=output, requires_grad=True, _node=node,
                         dtype=a.dtype,device=a.device)


    @staticmethod
    def pow(a, factor):
        output = jnp.power(a.tensor, factor)
        if not a.requires_grad:
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)

        node = Node(output, parents=(a.node,))
        node.backward = PowBackward(node, a, factor)
        return ll.Tensor(data=output, requires_grad=True, _node=node,device=a.device,
                         dtype=a.dtype)


    @staticmethod
    def multiple(a, b):
        output = a.tensor * b.tensor
        if not (a.requires_grad or b.requires_grad):
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)

        node = Node(output, parents=(a.node, b.node))
        node.backward = MulBackward(node, a, b)
        return ll.Tensor(data=output, requires_grad=True, _node=node,device=a.device,
                         dtype=a.dtype)


    @staticmethod
    def divide(a, b):
        output = a.tensor / b.tensor
        if not (a.requires_grad or b.requires_grad):
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)

        node = Node(output, parents=(a.node, b.node))
        node.backward = DivBackward(node, a, b)
        return ll.Tensor(data=output, requires_grad=True, _node=node,
                         dtype=a.dtype,device=a.device)

    @staticmethod
    def exp (a):
        output = jnp.exp(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        node = Node(output,parents=(a.node,))

        node.backward = expbackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def tanh (a) :
        output = jnp.tanh(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        node = Node(output,parents=(a.node,))

        node.backward = TanhBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod 
    def log (a) :
        output = jnp.log(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,device=a.device,dtype=a.dtype)
        node = Node(output,parents=(a.node,))
        node.backward = LogBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)

    @staticmethod
    def matmul (a,b,transpose_a=False,transpose_b=False) :
        a_tensor = jnp.swapaxes(a.tensor,-2,-1) if transpose_a else a.tensor
        b_tensor = jnp.swapaxes(b.tensor,-2,-1) if transpose_b else b.tensor
        output = jnp.matmul(a_tensor,b_tensor)

        if not (a.requires_grad or b.requires_grad) :
            return ll.Tensor(data=output,device=a.device,dtype=a.dtype)
        
        node = Node(output,parents=(a.node,b.node))
        
        node.backward = MatMulBackward(node=node,a=a,b=b,transpose_a=transpose_a,transpose_b=transpose_b)
        return ll.Tensor(data=output,requires_grad=True,_node=node,
                         device=a.device,dtype=a.dtype)
    
    @staticmethod
    def sum (a,axis=None,keepdims=False) :
        output = jnp.sum(a.tensor,axis=axis,keepdims=keepdims)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))

        node.backward = SumBackward(node,a,axis=axis,keepdims=keepdims)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def reshape (a,newshape) :
        output = jnp.reshape(a.tensor,newshape)
        if not a.requires_grad :
            return ll.Tensor(data=output,device=a.device,dtype=a.dtype)
        
        node = Node(output,parents=(a.node,))
        node.backward = ReshapeBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def transpose (a,axes=None) :
        output = jnp.transpose(a.tensor,axes=axes)
        if not a.requires_grad :
            return ll.Tensor(data=output,device=a.device,dtype=a.dtype)
        
        node = Node(output,parents=(a.node,))
   
        node.backward = TransposeBackward(node,a,axes=axes)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def flatten (a) :
        output = jnp.reshape(a.tensor,-1)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = FlattenBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         device=a.device,dtype=a.dtype)
    
    @staticmethod
    def getitem (a,idx) :
        output = a.tensor[idx]
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = GetItemBackward(node,a,idx)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         device=a.device,dtype=a.dtype)
    
    @staticmethod
    def max (a,axis=None,keepdims=False) :
        output = jnp.max(a.tensor,axis=axis,keepdims=keepdims)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = MaxBackward(node,a,axis=axis,keepdims=keepdims)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         device=a.device,dtype=a.dtype)
    
    @staticmethod
    def min (a,axis=None,keepdims=False) :
        output = jnp.min(a.tensor,axis=axis,keepdims=keepdims)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = MinBackward(node,a,axis=axis,keepdims=keepdims)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def where (condition,a,b) :
        output = jnp.where(condition.tensor,a.tensor,b.tensor)
        if not (a.requires_grad or b.requires_grad) :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,b.node))
        node.backward = WhereBackward(node,condition,a,b)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def abs (a) :
        output = jnp.abs(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))

        node.backward = AbsBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def clip (a,min_value,max_value) :
        output = jnp.clip(a.tensor,min_value,max_value)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = ClipBackward(node,a,min_value,max_value)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def expand_dims (a,axis) :
        output = jnp.expand_dims(a.tensor,axis=axis)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = ExpandDimsBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def squeeze (a,axis=None) :
        output = jnp.squeeze(a.tensor,axis=axis)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = SqueezeBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def unsqueeze (a,axis) :
        output = jnp.expand_dims(a.tensor,axis=axis)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = UnsqueezeBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def gelu_tanh (a) :
        """Gaussian Error Linear Unit activation function using tanh approximation."""
        sqrt_2_over_pi = jnp.sqrt(2 / jnp.pi)
        cdf = 0.5 * (1.0 + jnp.tanh(sqrt_2_over_pi * (a.tensor + 0.044715 * jnp.power(a.tensor, 3))))
        output = a.tensor * cdf

        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)

        node = Node(output, parents=(a.node,))

        node.backward = GeluTanhBackward(node,a)
        return ll.Tensor(data=output, requires_grad=True, _node=node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def softplus (a) :
        output = jnp.log1p(jnp.exp(a.tensor))
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = SoftplusBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def celu (a,alpha=1.0) :
        output = jnp.where(a.tensor > 0, a.tensor, alpha * (jnp.exp(a.tensor / alpha) - 1))
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = CeluBackward(node,a,alpha)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def selu (a,lambda_=1.0507,alpha=1.673) :
        output = jnp.where(a.tensor > 0, lambda_ * a.tensor, lambda_ * alpha * (jnp.exp(a.tensor) - 1))
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))

        node.backward = SeluBackward(node,a,lambda_=lambda_,alpha=alpha)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def softsign (a) :
        output = a.tensor / (1 + jnp.abs(a.tensor))
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = SoftsignBackward(node,a)
        
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod 
    def log_softmax(a,axis=None,keepdims=False) :
        shifted_logits = a.tensor - jnp.max(a.tensor, axis=axis, keepdims=True)
        logsumexp = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=axis, keepdims=True))
        output = shifted_logits - logsumexp

        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)

        node = Node(output, parents=(a.node,))
        node.backward = LogSoftmaxBackward(node,a,axis=axis,output=output,
                                           keepdims=keepdims)
        return ll.Tensor(data=output, requires_grad=True, _node=node,
                         dtype=a.dtype)
    
    @staticmethod
    def softmax (x,axis=None,keepdims=False,with_crossentropy=False) :
        max_x = jnp.max(x.tensor, axis=axis, keepdims=True)
        shifted_x = x.tensor - max_x
        exp_shifted = jnp.exp(shifted_x)
        sum_exp = jnp.sum(exp_shifted, axis=axis, keepdims=True)
        sum_exp = jnp.where(sum_exp==0,1e-6,sum_exp)
        output = exp_shifted / sum_exp

        if not x.requires_grad :
            return ll.Tensor(data=output,dtype=x.dtype,device=x.device)

        node = Node(output, parents=(x.node,))

        node.backward = SoftmaxBackward(
            node=node,x=x,axis=axis,
            with_crossentropy=with_crossentropy,output=output,
            keepdims=keepdims
        )
        return ll.Tensor(data=output, requires_grad=True, _node=node,
                         device=x.device,dtype=x.dtype)
    
    @staticmethod
    def pad (a,pad_width,mode='constant',constant_values=0) :
        output = jnp.pad(a.tensor,pad_width,mode=mode,constant_values=constant_values)
        if not a.requires_grad :
            return ll.Tensor(data=output,device=a.device,dtype=a.dtype)
        
        node = Node(output,parents=(a.node,))
        node.backward = PadBackward(node,a,pad_width=pad_width)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def broadcast_to (a,shape) :
        output = jnp.broadcast_to(a.tensor,shape)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = BroadcastToBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod 
    def elu (a,alpha=1.0) :
        output = jnp.where(a.tensor > 0, a.tensor, alpha * (jnp.exp(a.tensor) - 1))
        if not a.requires_grad :
            return ll.Tensor(data=output,device=a.device,dtype=a.dtype)
        
        node = Node(output,parents=(a.node,))
        node.backward = EluBackward(node,a,alpha)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def erf (a) :
        output = jnp.erf(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,device=a.device,dtype=a.dtype)
        
        node = Node(output,parents=(a.node,))
        node.backward = ErfBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         device=a.device,dtype=a.dtype)
    
    @staticmethod
    def sinh (a) :
        output = jnp.sinh(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = SinhBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def cosh (a) :
        output = jnp.cosh(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = CoshBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def tan (a) :
        output = jnp.tan(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = TanBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def sumproduct (a,b) :
        output = jnp.sum(a.tensor * b.tensor)
        if not (a.requires_grad or b.requires_grad) :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,b.node))
        node.backward = SumProductBackward(node,a,b)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def log10 (a) :
        output = jnp.log10(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
       
        node.backward = Log10Backward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def sqrt (a) :
        output = jnp.sqrt(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = SqrtBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def log2 (a) :
        output = jnp.log2(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,device=a.device,dtype=a.dtype)
        
        node = Node(output,parents=(a.node,))

        node.backward = Log2Backward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         device=a.device,dtype=a.dtype)
    
    @staticmethod
    def floor (a) :
        output = jnp.floor(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = FloorBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def ceil (a) :
        output = jnp.ceil(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = CeilBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def rint (a) :
        output = jnp.rint(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))

        node.backward = RintBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def reciprocal (a) :
        output = 1 / a.tensor
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = ReciprocalBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def sign (a) :
        output = jnp.sign(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = SignBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def identity (a) :
        output = a.tensor
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = IdentityBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def relu6 (a) :
        output = jnp.minimum(jnp.maximum(a.tensor, 0), 6)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = Relu6Backward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def hardshrink (a,lambda_=0.5) :
        output = jnp.where(jnp.abs(a.tensor) > lambda_, a.tensor, 0)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = HardShrinkBackward(node,a,lambda_)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def hardtanh (a,min_val=-1.0,max_val=1.0) :
        output = jnp.clip(a.tensor, min_val, max_val)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = HardTanhBackward(node,a,min_val,max_val)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def log1p (a) :
        output = jnp.log1p(a.tensor)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = Log1pBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def hardsigmoid (a) :
        output = jnp.clip((a.tensor + 1) / 2, 0, 1)
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = HardSigmoidBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)

    @staticmethod
    def hardmax (a,axis=None,keepdims=False) :
        max_indices = jnp.argmax(a.tensor, axis=axis, keepdims=True)
        output = jnp.zeros_like(a.tensor).at[max_indices].set(1)

        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)

        node = Node(output, parents=(a.node,))

        node.backward = HardMaxBackward(node,a,axis,keepdims)
        return ll.Tensor(data=output, requires_grad=True, _node=node,dtype=a.dtype,
                         device=a.device)
    
    @staticmethod 
    def logsigmoid (a) :
        output = -jnp.log1p(jnp.exp(-a.tensor))
        if not a.requires_grad :
            return ll.Tensor(data=output)
        
        node = Node(output,parents=(a.node,))
        node.backward = LogSigmoidBackward(node,a)
        return ll.Tensor(data=output,requires_grad=True,_node = node)
    
    @staticmethod
    def concat(tensors, axis=0):
        out = jnp.concatenate([t.tensor for t in tensors], axis=axis)

        requires_grad = any(t.requires_grad for t in tensors)
        if not requires_grad:
            return ll.Tensor(out, dtype=tensors[0].dtype, device=tensors[0].device)

        node = Node(out, parents=tuple(t.node for t in tensors))
        node.backward = ConcatBackward(node,axis)

        return ll.Tensor(out, dtype=tensors[0].dtype, device=tensors[0].device,
                        requires_grad=True, _node=node)


    @staticmethod
    def sigmoid(a):
        output = 1 / (1 + jnp.exp(-a.tensor))
        if not a.requires_grad:
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)

        node = Node(output, parents=(a.node,))

        node.backward = SigmoidBackward(node,a,output)
        return ll.Tensor(data=output, requires_grad=True, _node=node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod 
    def logsumexp(a, axis=None, keepdims=False):
        max_a = jnp.max(a.tensor, axis=axis, keepdims=True)
        shifted_a = a.tensor - max_a
        sum_exp = jnp.sum(jnp.exp(shifted_a), axis=axis, keepdims=True)
        output = max_a + jnp.log(sum_exp)

        if not a.requires_grad:
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)

        node = Node(output, parents=(a.node,))

        node.backward = LogSumExpBackward(node,a,axis,keepdims=keepdims,shifted_a=shifted_a,sum_exp=sum_exp)
        return ll.Tensor(data=output, requires_grad=True, _node=node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod
    def swish (a,beta=1.0) :
        sigmoid = 1 / (1 + jnp.exp(-beta * a.tensor))
        output = a.tensor * sigmoid
        if not a.requires_grad :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,))
        node.backward = SwishBackward(node,a,sigmoid,beta)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod 
    def logaddexp (a,b) :
        max_ab = jnp.maximum(a.tensor, b.tensor)
        output = max_ab + jnp.log(jnp.exp(a.tensor - max_ab) + jnp.exp(b.tensor - max_ab))
        if not (a.requires_grad or b.requires_grad) :
            return ll.Tensor(data=output,dtype=a.dtype,device=a.device)
        
        node = Node(output,parents=(a.node,b.node))
        node.backward = LogAddExpBackward(node,a,b)
        return ll.Tensor(data=output,requires_grad=True,_node = node,
                         dtype=a.dtype,device=a.device)
    
    @staticmethod 
    def relu (x) :
        output = jnp.maximum(0,x.tensor)

        if not x.requires_grad :
            return ll.Tensor(output,dtype=x.dtype,device=x.device)

        node = Node(output,parents=(x.node,))

    
        node.backward =ReLUBackward(node,x)
        return ll.Tensor(data=output,requires_grad=True,_node=node,
                         dtype=x.dtype,device=x.device)

    @staticmethod
    def leaky_relu(x ,alpha: float = 1e-3) :
        output = jnp.where(x.tensor > 0,x.tensor,x.tensor * alpha)

        if not x.requires_grad :
            return ll.Tensor(output,dtype=x.dtype,device=x.device)
        
        node = Node(output,parents=(x.node,))
        
        node.backward = LeakyReLUBackward(node,x,alpha)

        return ll.Tensor(output,requires_grad=True,_node=node,
                         dtype=x.dtype,device=x.device)
    
    @staticmethod
    def mean(x,axis=None,keepdims=False) :
        output = jnp.mean(x.tensor,axis=axis,keepdims=keepdims)
        
        if not x.requires_grad :
            return ll.Tensor(output,dtype=x.dtype,device=x.device)
        
        node = Node(data=output,parents=(x.node,))
       
       
        node.backward = MeanBackward(node,x,axis,keepdims)
        return ll.Tensor(data=output,requires_grad=True,_node=node,
                         dtype=x.dtype,device=x.device)
    
    @staticmethod 
    def var(x,axis=None,keepdims=False) :
        mean = jnp.mean(x.tensor,axis=axis,keepdims=keepdims)
        var = jnp.mean(jnp.power((x.tensor - mean),2),axis=axis,keepdims=keepdims)

        if not x.requires_grad :
            return ll.Tensor(var,dtype=x.dtype,device=x.device)

        node = Node(data=var,parents=(x.node,))
        node.backward = VarBackward(node,x,mean,axis,keepdims)
        return ll.Tensor(var,requires_grad=True,_node=node,
                         dtype=x.dtype,device=x.device)
    
    @staticmethod 
    def std (x,axis=None,keepdims=False,epsilon : float = 1e-6) :
        mean = jnp.mean(x.tensor,axis=axis,keepdims=keepdims)
        var = jnp.mean(jnp.power((x.tensor - mean),2),axis=axis,keepdims=keepdims)
        std = jnp.sqrt(var + epsilon)

        if not x.requires_grad :
            return ll.Tensor(std,dtype=x.dtype,device=x.device)

        node = Node(data=std,parents=(x.node,))
        node.backward = StdBackward(node,x,mean,std,axis,keepdims)
        return ll.Tensor(data=std,requires_grad=True,_node=node,
                         dtype=x.dtype,device=x.device)
    
    @staticmethod
    def conv2d(a, w, b=None, stride=1, pad=0):
        output, cols, x_padded = conv2d_forward(a.tensor, w.tensor, 
                                                b.tensor if b else None,
                                                stride, pad)
        if not (a.requires_grad or w.requires_grad) :
            return ll.Tensor(output,
                             dtype=a.dtype,device=a.device)

        
        parents = (a.node, w.node) + ((b.node,) if b else ())
        
        node = Node(output, parents=parents)
        node.backward = Conv2DBackward(
            a, w, b,
            stride, pad,
            cols, x_padded,
            output.shape,
            node
        )
        
        return ll.Tensor(data=output, requires_grad=True, _node=node,
                         dtype=a.dtype,device=a.device)


    @staticmethod
    def mse_loss (y_train,y_pred) :
        output = jnp.mean(jnp.power(y_pred.tensor - y_train,2))

        if not y_pred.requires_grad :
            return ll.Tensor(output,device=y_pred.device,dtype=y_pred.dtype)
        
        node = Node(data=output,parents=(y_pred.node,))

        node.backward = MSELossBackward(node,y_pred,y_train) 

        return ll.Tensor(data=output,requires_grad=True,dtype=y_pred.dtype,_node=node,
                         device=y_pred.device)
    
    @staticmethod 
    def mae_loss (y_train,y_pred) :
        output = jnp.mean(jnp.abs(y_pred.tensor - y_train))

        if not y_pred.requires_grad :
            return ll.Tensor(output,dtype=y_pred.dtype,device=y_pred.device)
        
        node = Node(data=output,parents=(y_pred.node,))
     
      
        node.backward = MAELossBackward(node,y_pred,y_train)

        return ll.Tensor(output,dtype=y_pred.dtype,requires_grad=True,_node=node,
                         device=y_pred.device)
    
    @staticmethod 
    def huber_loss (y_train,y_pred ,delta=1.0) :
        loss = y_pred.tensor - y_train 

        output = jnp.where(
            jnp.abs(loss) <= delta, 
            0.5 * jnp.mean(jnp.power(loss,2)),
            delta * jnp.abs(loss) * -(0.5 * delta)
        )

        if not y_pred.requires_grad :
            return ll.Tensor(output,dtype=y_pred.dtype,device=y_pred.device)
        node = Node(data=output,parents=(y_pred.node))
        node.backward = HuberLossBackward(node,y_pred,y_train,delta,loss) 
        return ll.Tensor(output,requires_grad=True,_node=node,
                         dtype=y_pred.dtype,device=y_pred.device)
    
    @staticmethod
    def bce_loss(y_train, y_pred, epsilon=1e-5):
        pred = jnp.clip(y_pred.tensor, epsilon, 1 - epsilon)

        output = (-1 / y_train.shape[0]) * jnp.sum(
            y_train * jnp.log(pred) +
            (1 - y_train) * jnp.log(1 - pred)
        )

        if not y_pred.requires_grad:
            return ll.Tensor(output,dtype=y_pred.dtype,device=y_pred.device)

        node = Node(data=output, parents=(y_pred.node,))
        node.backward = BCELossBackward(node, y_pred, y_train, epsilon)

        return ll.Tensor(output, dtype=y_pred.dtype, requires_grad=True, _node=node,
                         device=y_pred.device)

    @staticmethod
    def cross_entropy(y_true, logits):
        orig_shape = logits.shape
        num_classes = orig_shape[-1]

        flat_logits = logits.tensor.reshape(-1, num_classes)
        flat_y = y_true.reshape(-1, num_classes)
        N = flat_y.shape[0]


        log_probs = jax.nn.log_softmax(flat_logits, axis=-1)

        output = -jnp.sum(flat_y * log_probs) / N

        if not logits.requires_grad:
            return ll.Tensor(output,dtype=logits.dtype,device=logits.device)

        node = Node(data=output, parents=(logits.node,))
        node.backward = CrossEntropyBackward(node, logits, y_true)

        return ll.Tensor(output, dtype=logits.dtype, requires_grad=True, _node=node,
                         device=logits.device)

    @staticmethod
    def sparse_cross_entropy(y_true, logits):
        orig_shape = logits.shape
        num_classes = orig_shape[-1]

        flat_logits = logits.tensor.reshape(-1, num_classes)
        flat_labels = jnp.array(y_true).reshape(-1)
        N = flat_labels.shape[0]

        log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
        picked = log_probs[jnp.arange(N), flat_labels]

        output = -jnp.mean(picked)

        if not logits.requires_grad:
            return ll.Tensor(output,dtype=logits.dtype,device=logits.device)

        node = Node(data=output, parents=(logits.node,))
        node.backward = SparseCrossEntropyBackward(node, logits, y_true)

        return ll.Tensor(output, dtype=logits.dtype, requires_grad=True, _node=node,
                         device=logits.device)

    @staticmethod
    def masked_cross_entropy(y_true, logits, mask):
        logits_val = logits.tensor
        y_val = y_true
        mask_val = mask

        orig_shape = logits_val.shape
        num_classes = orig_shape[-1]

        flat_logits = logits_val.reshape(-1, num_classes)
        flat_y = y_val.reshape(-1, num_classes)
        flat_mask = mask_val.reshape(-1)

        log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
        token_losses = -jnp.sum(flat_y * log_probs, axis=-1)

        masked_loss = jnp.sum(token_losses * flat_mask)
        total = jnp.sum(flat_mask) + 1e-8
        output = masked_loss / total

        if not logits.requires_grad:
            return ll.Tensor(output,dtype=logits.dtype,device=logits.device)

        node = Node(data=output, parents=(logits.node,))
        node.backward = MaskedCrossEntropyBackward(node, logits, y_true, mask)

        return ll.Tensor(output, dtype=logits.dtype, requires_grad=True, _node=node,
                         device=logits.device)

    @staticmethod
    def masked_sparse_cross_entropy(y_true, logits, mask):
        logits_val = logits.tensor
        labels = jnp.array(y_true).reshape(-1)
        mask_val = mask.reshape(-1)

        orig_shape = logits_val.shape
        num_classes = orig_shape[-1]

        flat_logits = logits_val.reshape(-1, num_classes)
        log_probs = jax.nn.log_softmax(flat_logits, axis=-1)

        picked = log_probs[jnp.arange(labels.shape[0]), labels]
        masked_loss = -jnp.sum(picked * mask_val)

        total = jnp.sum(mask_val) + 1e-8
        output = masked_loss / total

        if not logits.requires_grad:
            return ll.Tensor(output,dtype=logits.dtype,device=logits.device)

        node = Node(data=output, parents=(logits.node,))
        node.backward = MaskedSparseCrossEntropyBackward(node, logits, y_true, mask)

        return ll.Tensor(output, dtype=logits.dtype, requires_grad=True, _node=node,
                         device=logits.device)

    @staticmethod
    def accuracy(y_true, logits):
        preds = jnp.argmax(logits.tensor, axis=-1)
        labels = jnp.argmax(y_true, axis=-1)
        acc = jnp.mean(preds == labels)
        return acc

    @staticmethod
    def sparse_accuracy(y_true, logits):
        preds = jnp.argmax(logits.tensor, axis=-1)
        labels = y_true
        acc = jnp.mean(preds == labels)
        return acc

    @staticmethod
    def masked_accuracy(y_true, logits, mask):
        preds = jnp.argmax(logits.tensor, axis=-1)
        labels = jnp.argmax(y_true, axis=-1)
        correct = (preds == labels) * mask
        return jnp.sum(correct) / (jnp.sum(mask) + 1e-8)

    @staticmethod
    def masked_sparse_accuracy(y_true, logits, mask):
        preds = jnp.argmax(logits.tensor, axis=-1)
        labels = y_true
        correct = (preds == labels) * mask
        return jnp.sum(correct) / (jnp.sum(mask) + 1e-8)

    @staticmethod
    def maxpool2d(x, kernel=(2,2), stride=(2,2), padding=0):
        Kh, Kw = kernel
        Sh, Sw = stride
        pad = padding

        input = x.tensor
        N, H, W, C = input.shape

        H_out = (H + 2 * pad - Kh) // Sh + 1
        W_out = (W + 2 * pad - Kw) // Sw + 1

        output = jnp.zeros((N, H_out, W_out, C))
        argmax_mask = jnp.zeros((N, H_out, W_out, C, Kh, Kw), dtype=bool)

        padded = jnp.pad(input, ((0,0),(pad,pad),(pad,pad),(0,0)), mode="constant")

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * Sh
                w_start = j * Sw
                h_end = h_start + Kh
                w_end = w_start + Kw

                window = padded[:, h_start:h_end, w_start:w_end, :]   

                max_vals = jnp.max(window, axis=(1,2))
                output = output.at[:, i, j, :].set(max_vals)

                eq_mask = (window == max_vals[:, None, None, :])   
                argmax_mask = argmax_mask.at[:, i, j].set(jnp.moveaxis(eq_mask, [1,2,3], [2,3,1]))

        if not x.requires_grad:
            return ll.Tensor(output,device=x.device,dtype=x.dtype)

        node = Node(data=output, parents=(x.node,))
        node.backward = MaxPool2DBackward(node, x, kernel, stride, padding, argmax_mask)

        return ll.Tensor(output, dtype=x.dtype, requires_grad=True, _node=node,
                         device=x.device)

    @staticmethod
    def lstm_cell(x_t, h_prev, c_prev, w_gate: tuple, wh_gate: tuple):

        
        i = 1 / (1 + jnp.exp(-(jnp.matmul(x_t.tensor,  w_gate[0].tensor) +
                            jnp.matmul(h_prev.tensor, wh_gate[0].tensor))))

        f = 1 / (1 + jnp.exp(-(jnp.matmul(x_t.tensor,  w_gate[1].tensor) +
                            jnp.matmul(h_prev.tensor, wh_gate[1].tensor))))

        g = jnp.tanh(jnp.matmul(x_t.tensor,  w_gate[2].tensor) +
                    jnp.matmul(h_prev.tensor, wh_gate[2].tensor))

        output = f * c_prev.tensor + i * g 

        if not x_t.requires_grad : 
            return ll.Tensor(output,dtype=x_t.dtype,device=x_t.device)
        
        node = Node(
            data=output,parents=(x_t.node,h_prev.node,c_prev.node,
                                 w_gate[0].node,w_gate[1].node,w_gate[2].node,
                                 wh_gate[0].node,wh_gate[1].node,wh_gate[2].node)
        )
        node.backward = lstm_cell_backward(node,
                                           i=i,f=f,g=g,x_t=x_t,
                                           h_prev=h_prev,c_prev=c_prev,Wi=w_gate[0],Wf=w_gate[1],Wg=w_gate[2],
                                           Ui=wh_gate[0],Uf=wh_gate[1],Ug=wh_gate[2])
        return ll.Tensor(output,dtype=x_t.dtype,device=x_t.device,requires_grad=True,_node=node)
    
    @staticmethod

    def lstm_hidden (x_t,h_prev,cell_state,w_output,wh_output) :
        o = 1 / (1 + jnp.exp(-(jnp.matmul(x_t.tensor,w_output.tensor) + jnp.matmul(h_prev.tensor,wh_output.tensor))))
        hc =   jnp.tanh(cell_state.tensor)
        output = o * hc 

        if not x_t.requires_grad :
            return ll.Tensor(data=output,dtype=x_t.dtype,device=x_t.device)
        
        node = Node(data=output,parents=(
           x_t.node,h_prev.node,cell_state.node,w_output.node,wh_output.node
        ))
        node.backward = lstm_hidden_backward(
            node,o=o,x_t=x_t,hc=hc,h_prev=h_prev,cell_state=cell_state,
            w_output=w_output,wh_output=wh_output
        )
        return ll.Tensor(
            data=output,dtype=x_t.dtype,device=x_t.device,requires_grad=True,_node=node
        )
    
    @staticmethod
    def sin (x) :
        output = jnp.sin(x.tensor)

        if not x.requires_grad :
            return ll.Tensor(x,dtype=x.dtype,
                             device=x.device)
        
        node = Node(
            data=output,
            parents=(x.node,)

        )
        node.backward = SinBackward(
            node,x
        )
        return ll.Tensor(output,dtype=x.dtype,device=x.device,requires_grad=True,_node=node)
    
    @staticmethod
    def cos (x) :
        output = jnp.cos(x.tensor)

        if not x.requires_grad :
            return ll.Tensor(data=output,dtype=x.dtype,device=x.device)
        
        node = Node(output,parents=(x.node,))
        node.backward = CosBackward(node,x)
        return ll.Tensor(data=output,dtype=x.dtype,device=x.device,requires_grad=True,_node=node)

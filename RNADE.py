'''
Class that builds the graph for the Real Neural Autoregressive Distribution Estimator(RNADE)
'''
import numpy
import theano
import theano.tensor as T
import pdb
from theano.compat.python2x import OrderedDict

def shared_normal(shape, scale=1,name=None):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    return theano.shared(name=name,value=numpy.random.normal(
    scale=scale, size=shape).astype(theano.config.floatX))

def shared_zeros(shape,name=None):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(name=name,value=numpy.zeros(shape, dtype=theano.config.floatX))

def sigmoid(x):
  return 0.5*numpy.tanh(0.5*x) + 0.5

def constantX(value):
    """
    Returns a constant of value `value` with floatX dtype
    """
    return theano.tensor.constant(numpy.asarray(value, dtype=theano.config.floatX))

def log_sum_exp(x, axis=1):
    max_x = T.max(x, axis)
    return max_x + T.log(T.sum(T.exp(x - T.shape_padright(max_x, 1)), axis))

def sigmoid(s):
    s = 1.0 / (1.0 + numpy.exp(-1.0 * s))
    return s

activation_function = {}
activation_function['ReLU'] = lambda x: numpy.maximum(0.,x)
activation_function['sigmoid'] = lambda x: sigmoid(x)

def softmax(X):
    """Calculates softmax row-wise"""
    if X.ndim == 1:
        X = X - numpy.max(X)
        e = numpy.exp(X)
        return e/e.sum()
    else:
        X = X - numpy.max(X,1)[:,numpy.newaxis]
        e = numpy.exp(X)    
        return e/e.sum(1)[:,numpy.newaxis]

def random_component(component_probabilities):
    r = numpy.random.random(1)[0]
    accum = 0.0
    for i,p in enumerate(component_probabilities):
        accum += p
        if r <= accum:
            return i

floatX = theano.config.floatX

class RNADE:
    def __init__(self,n_visible,n_hidden,n_components,hidden_act='ReLU',l2=0.):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_components = n_components
        self.W = shared_normal((n_visible, n_hidden), 0.01,'W')
        self.b_alpha = shared_normal((n_visible,n_components),0.01,'b_alpha')
        self.V_alpha = shared_normal((n_visible,n_hidden,n_components),0.01,'V_alpha')
        self.b_mu = shared_normal((n_visible,n_components),0.01,'b_mu')
        self.V_mu = shared_normal((n_visible,n_hidden,n_components),0.01,'V_mu')
        #from Benigno's implemenation
        self.b_sigma = shared_normal((n_visible,n_components),0.01,'b_sigma')
        self.b_sigma.set_value(self.b_sigma.get_value() + 1.0)
        self.V_sigma = shared_normal((n_visible,n_hidden,n_components),0.01,'V_sigma')
        #Initialising activation rescaling to all 1s. 
        self.activation_rescaling = shared_zeros((n_visible,),'activation_rescaling')
        self.activation_rescaling.set_value(self.activation_rescaling.get_value() + 1.0)
        self.params = [self.W,self.b_alpha,self.V_alpha,self.b_mu,self.V_mu,self.b_sigma,self.V_sigma,self.activation_rescaling]
        self.hidden_act = hidden_act
        self.l2 = l2
        if self.hidden_act == 'sigmoid':
            self.nonlinearity = T.nnet.sigmoid
        elif self.hidden_act == 'ReLU':
            self.nonlinearity = lambda x:T.maximum(x,0.)
            #self.nonlinearity = lambda x: x * (x > 0)
        self.v = T.matrix('v')
        #self.build_fprop()
        #self.build_fprop_two()
        #self.build_gradients()
        #self.build_cost()
    
    def sym_logdensity(self, x):
        """ x is a matrix of column datapoints (VxB) V = n_visible, B = batch size """
        def density_given_previous_a_and_x(x, w, V_alpha, b_alpha, V_mu, b_mu, V_sigma, b_sigma,activation_factor, p_prev, a_prev, x_prev,):
            a = a_prev + T.dot(T.shape_padright(x_prev, 1), T.shape_padleft(w, 1))
            h = self.nonlinearity(a * activation_factor)  # BxH

            Alpha = T.nnet.softmax(T.dot(h, V_alpha) + T.shape_padleft(b_alpha))  # BxC
            Mu = T.dot(h, V_mu) + T.shape_padleft(b_mu)  # BxC
            Sigma = T.exp((T.dot(h, V_sigma) + T.shape_padleft(b_sigma)))  # BxC
            p = p_prev + log_sum_exp(-constantX(0.5) * T.sqr((Mu - T.shape_padright(x, 1)) / Sigma) - T.log(Sigma) - constantX(0.5 * numpy.log(2 * numpy.pi)) + T.log(Alpha))
            return (p, a, x)
        # First element is different (it is predicted from the bias only)
        a0 = T.zeros_like(T.dot(x.T, self.W))  # BxH
        p0 = T.zeros_like(x[0])
        x0 = T.ones_like(x[0])
        ([ps, _as, _xs], updates) = theano.scan(density_given_previous_a_and_x,
                                                sequences=[x, self.W, self.V_alpha, self.b_alpha, self.V_mu, self.b_mu, self.V_sigma, self.b_sigma,self.activation_rescaling],
                                                outputs_info=[p0, a0, x0])
        return (ps[-1], updates)

    def build_fprop(self,):
        print 'Using theano grads.'
        self.ps,updates = self.sym_logdensity(self.v.T)
        self.cost = -T.mean(self.ps,axis=0) + self.l2*T.sum(self.W**2)
        #self.fprop = theano.function([self.v],self.ps)

    def build_fprop_two(self,):
        print 'Using custom grads.'
        self.ps,self.grads,updats = self.sym_gradients_new(self.v.T)
        self.cost = T.mean(self.ps,axis=0) + self.l2*T.sum(self.W**2)#no negative sign! 
        #self.fprop = theano.function([self.v],self.ps)

    def sym_gradients_new(self, X):
        #non_linearity_name = self.parameters["nonlinearity"].get_name()
        #assert(non_linearity_name == "sigmoid" or non_linearity_name == "RLU")
        # First element is different (it is predicted from the bias only)
        init_a = T.zeros_like(T.dot(X.T, self.W))  # BxH
        init_x = T.ones_like(X[0])

        def a_i_given_a_im1(x, w, a_prev, x_prev):
            a = a_prev + T.dot(T.shape_padright(x_prev, 1), T.shape_padleft(w, 1))
            return (a, x)
        ([As, _], updates) = theano.scan(a_i_given_a_im1, sequences=[X, self.W], outputs_info=[init_a, init_x])
        top_activations = As[-1]
        Xs_m1 = T.set_subtensor(X[1:, :], X[0:-1, :])
        Xs_m1 = T.set_subtensor(Xs_m1[0, :], 1)

        # Reconstruct the previous activations and calculate (for that visible dimension) the density and all the gradients
        def density_and_gradients(x_i, x_im1, w_i, V_alpha, b_alpha, V_mu, b_mu, V_sigma, b_sigma, activation_factor, a_i, lp_accum, dP_da_ip1):
            B = T.cast(x_i.shape[0], floatX)
            pot = a_i * activation_factor
            h = self.nonlinearity(pot)  # BxH

            z_alpha = T.dot(h, V_alpha) + T.shape_padleft(b_alpha)
            z_mu = T.dot(h, V_mu) + T.shape_padleft(b_mu)
            z_sigma = T.dot(h, V_sigma) + T.shape_padleft(b_sigma)

            Alpha = T.nnet.softmax(z_alpha)  # BxC
            Mu = z_mu  # BxC
            Sigma = T.exp(z_sigma)  # BxC

            Phi = -constantX(0.5) * T.sqr((Mu - T.shape_padright(x_i, 1)) / Sigma) - T.log(Sigma) - constantX(0.5 * numpy.log(2 * numpy.pi))
            wPhi = T.maximum(Phi + T.log(Alpha), constantX(-100.0))

            lp_current = -log_sum_exp(wPhi)  # negative log likelihood
            # lp_current_sum = T.sum(lp_current)

            Pi = T.exp(wPhi - T.shape_padright(lp_current, 1))  # #
            dp_dz_alpha = Pi - Alpha  # BxC
            # dp_dz_alpha = T.grad(lp_current_sum, z_alpha)
            gb_alpha = dp_dz_alpha.mean(0, dtype=floatX)  # C
            gV_alpha = T.dot(h.T, dp_dz_alpha) / B  # HxC

            dp_dz_mu = -Pi * (Mu - T.shape_padright(x_i, 1)) / T.sqr(Sigma)
            # dp_dz_mu = T.grad(lp_current_sum, z_mu)
            dp_dz_mu = dp_dz_mu * Sigma  # Heuristic
            gb_mu = dp_dz_mu.mean(0, dtype=floatX)
            gV_mu = T.dot(h.T, dp_dz_mu) / B

            dp_dz_sigma = Pi * (T.sqr(T.shape_padright(x_i, 1) - Mu) / T.sqr(Sigma) - 1)
            # dp_dz_sigma = T.grad(lp_current_sum, z_sigma)
            gb_sigma = dp_dz_sigma.mean(0, dtype=floatX)
            gV_sigma = T.dot(h.T, dp_dz_sigma) / B

            dp_dh = T.dot(dp_dz_alpha, V_alpha.T) + T.dot(dp_dz_mu, V_mu.T) + T.dot(dp_dz_sigma, V_sigma.T)  # BxH
            if self.hidden_act == "sigmoid":
                dp_dpot = dp_dh * h * (1 - h)
            elif self.hidden_act == "ReLU":
                dp_dpot = dp_dh * (pot > 0)

            gfact = (dp_dpot * a_i).sum(1).mean(0, dtype=floatX)  # 1

            dP_da_i = dP_da_ip1 + dp_dpot * activation_factor  # BxH
            gW = T.dot(T.shape_padleft(x_im1, 1), dP_da_i).flatten() / B

            return (a_i - T.dot(T.shape_padright(x_im1, 1), T.shape_padleft(w_i, 1)),
                    lp_accum + lp_current,
                    dP_da_i,
                    gW, gb_alpha, gV_alpha, gb_mu, gV_mu, gb_sigma, gV_sigma, gfact)

        p_accum = T.zeros_like(X[0])
        dP_da_ip1 = T.zeros_like(top_activations)
        ([_, ps, _, gW, gb_alpha, gV_alpha, gb_mu, gV_mu, gb_sigma, gV_sigma, gfact], updates2) = theano.scan(density_and_gradients,
                                                go_backwards=True,
                                                sequences=[X, Xs_m1, self.W, self.V_alpha, self.b_alpha, self.V_mu, self.b_mu, self.V_sigma, self.b_sigma, self.activation_rescaling],
                                                outputs_info=[top_activations, p_accum, dP_da_ip1, None, None, None, None, None, None, None, None])
        # scan with go_backwards returns the matrices in the order they were created, so we have to reverse the order of the rows
        gW = gW[::-1, :]
        gb_alpha = gb_alpha[::-1, :]
        gV_alpha = gV_alpha[::-1, :, :]
        gb_mu = gb_mu[::-1, :]
        gV_mu = gV_mu[::-1, :, :]
        gb_sigma = gb_sigma[::-1, :]
        gV_sigma = gV_sigma[::-1, :, :]
        gfact = gfact[::-1]

        updates.update(updates2)  # Returns None
        return (ps[-1], {"W": gW, "b_alpha": gb_alpha, "V_alpha": gV_alpha, "b_mu": gb_mu, "V_mu": gV_mu, "b_sigma": gb_sigma, "V_sigma": gV_sigma, "activation_rescaling": gfact}, updates)
    
    def build_gradients(self,):
        grad_probs,grad_updates,update = self.sym_gradients_new(self.v.T)
        proper_updates = OrderedDict()
        for param in self.params:
            proper_updates[param] = param - 0.1 * grad_updates[param.name]
        self.test = theano.function([self.v],grad_probs,updates=proper_updates)
    
    def sample(self,n):
        W = self.W.get_value()
        V_alpha = self.V_alpha.get_value()
        b_alpha = self.b_alpha.get_value()
        V_mu = self.V_mu.get_value()
        b_mu = self.b_mu.get_value()
        V_sigma = self.V_sigma.get_value()
        b_sigma = self.b_sigma.get_value()
        activation_rescaling = self.activation_rescaling.get_value()
        samples = numpy.zeros((self.n_visible, n))
        for s in xrange(n):
            a = numpy.zeros((self.n_hidden,))  # H
            for i in xrange(self.n_visible):
                if i == 0:
                    a = W[i, :]
                else:
                    a = a + W[i, :] * samples[i - 1, s]
                h = activation_function[self.hidden_act](a * activation_rescaling[i])
                alpha = softmax(numpy.dot(h, V_alpha[i]) + b_alpha[i])  # C
                Mu = numpy.dot(h, V_mu[i]) + b_mu[i]  # C
                Sigma = numpy.minimum(numpy.exp(numpy.dot(h, V_sigma[i]) + b_sigma[i]), 1)
                comp = random_component(alpha)
                samples[i, s] = numpy.random.normal(Mu[comp], Sigma[comp])
        return samples.T


if __name__ == '__main__':
    n_visible = 10
    n_hidden = 20
    n_components = 2
    hidden_act = 'ReLU'
    test = RNADE(n_visible,n_hidden,n_components,hidden_act)
    samples = test.sample(20)
    #print 'Finding grads'
    #grads = T.grad(test.cost,test.params)
    #test_data = numpy.random.random((100,n_visible))
    #output = test.fprop(test_data)
    #pdb.set_trace()
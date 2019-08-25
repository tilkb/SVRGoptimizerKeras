from keras.optimizers import Optimizer
from keras import backend as K
from keras.legacy import interfaces

class SVRG(Optimizer):
    """Stochastic Variance Reduced Gradient optimizer

    # Arguments
        lr: float >= 0. Learning rate.
        decay: float >= 0. Learning rate decay over each update.
        mean_calculation_step: int > 0. Number of mean gradient calculation step.
        update_step: int > 0. Number of weight update step.

    #Reference
    - [Accelerating Stochastic Gradient Descent using Predictive Variance Reduction](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf)
    """
    def __init__(self, lr=0.01, decay=0., mean_calculation_step=20, update_step=20, **kwargs):
        super(SVRG, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.state_counter = K.variable(0, dtype='int64', name='state_counter')
        self.initial_decay = decay
        self.mean_calculation_step = mean_calculation_step
        self.update_step = update_step
        

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.updates.append(K.update_add(self.state_counter, 1))
        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))
        shapes = [K.int_shape(p) for p in params]
        grad_mean = [K.zeros(shape) for shape in shapes]
        prev_weights = [p for p in params]
        self.weights = [self.iterations] + grad_mean + prev_weights
        old_grads = self.get_gradients(loss, prev_weights)
        for p, g, g_mean,prev, old_g in zip(params, grads, grad_mean,prev_weights, old_grads):
            #update part
            grad = g+g_mean-old_g  
            v = -lr * grad 
            new_p = p + v
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            new_p =K.switch(self.state_counter>self.mean_calculation_step,new_p,p)
            self.updates.append(K.update(p, new_p))

            #statistics part  
            grad_stat = K.switch(self.state_counter<=self.mean_calculation_step,g*(1.0/self.mean_calculation_step), K.zeros_like(g))
            self.updates.append(K.update_add(g_mean, grad_stat))  
            #switch statistics --> update
            temp_params = K.switch(self.state_counter<=self.mean_calculation_step,p,prev)
            self.updates.append(K.update(prev,temp_params))
            #switch update --> statistics
            temp_g_mean = K.switch(K.equal(self.state_counter,self.mean_calculation_step+self.update_step),K.zeros_like(g_mean),g_mean)
            self.updates.append(K.update(g_mean,temp_g_mean))

        counter = K.switch(self.state_counter>self.mean_calculation_step+self.update_step,K.constant(0, dtype='int64'),self.state_counter)
        self.updates.append(K.update(self.state_counter, counter))
        return self.updates


    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay)),
                  'mean_calculation_step': self.mean_calculation_step,
                  'update_step': self.update_step}
        base_config = super(SVRG, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

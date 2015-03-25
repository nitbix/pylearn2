"""
A module containing different learning rules for use with the SGD training
algorithm.
"""
import numpy as np
import warnings

from theano.compat import six
from theano import config,shared
from theano import tensor as T
from theano.ifelse import ifelse

from pylearn2.compat import OrderedDict
from pylearn2.space import NullSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.monitor import Monitor


class LearningRule():
    """
    A pylearn2 learning rule is an object which computes new parameter values
    given (1) a learning rate (2) current parameter values and (3) the current
    estimated gradient.
    """

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        """
        Method called by the training algorithm, which allows LearningRules to
        add monitoring channels.

        Parameters
        ----------
        monitor : pylearn2.monitor.Monitor
            Monitor object, to which the rule should register additional
            monitoring channels.
        monitoring_dataset : pylearn2.datasets.dataset.Dataset or dict
            Dataset instance or dictionary whose values are Dataset objects.
        """
        pass

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        Provides the symbolic (theano) description of the updates needed to
        perform this learning rule.

        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.

        Returns
        -------
        updates : OrderdDict
            A dictionary mapping from the old model parameters, to their new
            values after a single iteration of the learning rule.

        Notes
        -----
        e.g. for standard SGD, one would return `sgd_rule_updates` defined
        below. Note that such a `LearningRule` object is not implemented, as
        these updates are implemented by default when the `learning_rule`
        parameter of sgd.SGD.__init__ is None.

        .. code-block::  python

            sgd_rule_updates = OrderedDict()
            for (param, grad) in grads.iteritems():
                sgd_rule_updates[k] = (param - learning_rate *
                                       lr_scalers.get(param, 1.) * grad)
        """
        raise NotImplementedError(str(type(self)) + " does not implement "
                                  "get_updates.")


class Momentum(LearningRule):
    """
    Implements momentum as described in Section 9 of
    "A Practical Guide to Training Restricted Boltzmann Machines",
    Geoffrey Hinton.

    Parameters are updated by the formula:
    inc := momentum * inc - learning_rate * d cost / d param
    param := param + inc

    Parameters
    ----------
    init_momentum : float
        Initial value for the momentum coefficient. It remains fixed during
        training unless used with a `MomentumAdjustor`
        extension.
    nesterov_momentum: bool
        Use the accelerated momentum technique described in:
        "Advances in Optimizing Recurrent Networks", Yoshua Bengio, et al.

    """

    def __init__(self, init_momentum, nesterov_momentum=False):
        assert init_momentum >= 0.
        assert init_momentum < 1.
        self.momentum = sharedX(init_momentum, 'momentum')
        self.nesterov_momentum = nesterov_momentum

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        """
        Activates monitoring of the momentum.

        Parameters
        ----------
        monitor : pylearn2.monitor.Monitor
            Monitor object, to which the rule should register additional
            monitoring channels.
        monitoring_dataset : pylearn2.datasets.dataset.Dataset or dict
            Dataset instance or dictionary whose values are Dataset objects.
        """
        monitor.add_channel(
            name='momentum',
            ipt=None,
            val=self.momentum,
            data_specs=(NullSpace(), ''),
            dataset=monitoring_dataset)

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        Provides the updates for learning with gradient descent + momentum.

        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.
        """

        updates = OrderedDict()

        for (param, grad) in six.iteritems(grads):
            vel = sharedX(param.get_value() * 0.)
            assert param.dtype == vel.dtype
            assert grad.dtype == param.dtype
            if param.name is not None:
                vel.name = 'vel_' + param.name

            scaled_lr = learning_rate * lr_scalers.get(param, 1.)
            updates[vel] = self.momentum * vel - scaled_lr * grad

            inc = updates[vel]
            if self.nesterov_momentum:
                inc = self.momentum * inc - scaled_lr * grad

            assert inc.dtype == vel.dtype
            updates[param] = param + inc

        return updates


class MomentumAdjustor(TrainExtension):
    """
    A TrainExtension that implements a linear momentum schedule.

    Parameters
    ----------
    final_momentum : float
        The momentum coefficient to use at the end of learning.
    start : int
        The epoch on which to start growing the momentum coefficient.
    saturate : int
        The epoch on which the moment should reach its final value.
    """
    def __init__(self, final_momentum, start, saturate):
        if saturate < start:
            raise TypeError("Momentum can't saturate at its maximum value " +
                            "before it starts increasing.")

        self.__dict__.update(locals())
        del self.self
        self._initialized = False
        self._count = 0

    def setup(self, model, dataset, algorithm):
        """
        Initializes the momentum schedule based on epochs_seen.

        Parameters
        ----------
        model : pylearn2.models.Model
            The model to which the training algorithm is applied.
        dataset : pylearn2.datasets.Dataset
            The dataset to which the model is applied.
        algorithm : pylearn2.training_algorithms.TrainingAlgorithm
            Describes how gradients should be updated.
        """
        monitor = Monitor.get_monitor(model)
        self._count = monitor.get_epochs_seen()
        self._apply_momentum(algorithm)

    def on_monitor(self, model, dataset, algorithm):
        """
        Updates the momentum according to the linear schedule.

        Parameters
        ----------
        model : pylearn2.models.Model
            The model to which the training algorithm is applied.
        dataset : pylearn2.datasets.Dataset
            The dataset to which the model is applied.
        algorithm : pylearn2.training_algorithms.TrainingAlgorithm
            Describes how gradients should be updated.
        """
        self._count += 1
        self._apply_momentum(algorithm)

    def _apply_momentum(self, algorithm):
        """Updates the momentum on algorithm based on the epochs elapsed."""
        if hasattr(algorithm, 'learning_rule'):
            momentum = algorithm.learning_rule.momentum
        else:
            # TODO: remove once training_algorithm.sgd.SGD(init_momentum)
            # is officially deprecated.
            momentum = algorithm.momentum

        if not self._initialized:
            self._init_momentum = momentum.get_value()
            self._initialized = True

        momentum.set_value(np.cast[config.floatX](self.current_momentum()))

    def current_momentum(self):
        """Returns the momentum currently desired by the schedule."""
        w = self.saturate - self.start

        if w == 0:
            # saturate=start, so just jump straight to final momentum
            if self._count >= self.start:
                return self.final_momentum
            return self._init_momentum

        alpha = float(self._count - self.start) / float(w)
        if alpha < 0.:
            alpha = 0.
        if alpha > 1.:
            alpha = 1.
        return self._init_momentum * (1 - alpha) + alpha * self.final_momentum


class AdaDelta(LearningRule):
    """
    Implements the AdaDelta learning rule as described in:
    "AdaDelta: An Adaptive Learning Rate Method", Matthew D. Zeiler.

    Parameters
    ----------
    decay : float, optional
        Decay rate :math:`\\rho` in Algorithm 1 of the aforementioned
        paper.
    """

    def __init__(self, decay=0.95):
        assert decay >= 0.
        assert decay < 1.
        self.decay = decay

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        Compute the AdaDelta updates

        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.
        """
        updates = OrderedDict()
        for param in grads.keys():

            # mean_squared_grad := E[g^2]_{t-1}
            mean_square_grad = sharedX(param.get_value() * 0.)
            # mean_square_dx := E[(\Delta x)^2]_{t-1}
            mean_square_dx = sharedX(param.get_value() * 0.)

            if param.name is not None:
                mean_square_grad.name = 'mean_square_grad_' + param.name
                mean_square_dx.name = 'mean_square_dx_' + param.name

            # Accumulate gradient
            new_mean_squared_grad = (
                self.decay * mean_square_grad +
                (1 - self.decay) * T.sqr(grads[param])
            )

            # Compute update
            epsilon = lr_scalers.get(param, 1.) * learning_rate
            rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
            rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
            delta_x_t = - rms_dx_tm1 / rms_grad_t * grads[param]

            # Accumulate updates
            new_mean_square_dx = (
                self.decay * mean_square_dx +
                (1 - self.decay) * T.sqr(delta_x_t)
            )

            # Apply update
            updates[mean_square_grad] = new_mean_squared_grad
            updates[mean_square_dx] = new_mean_square_dx
            updates[param] = param + delta_x_t

        return updates


class AdaGrad(LearningRule):
    """
    Implements the AdaGrad learning rule as described in:
    "Adaptive subgradient methods for online learning and
    stochastic optimization", Duchi J, Hazan E, Singer Y.
    """

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        Compute the AdaGrad updates

        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.
        """
        updates = OrderedDict()
        for param in grads.keys():

            # sum_square_grad := \sum g^2
            sum_square_grad = sharedX(param.get_value() * 0.)

            if param.name is not None:
                sum_square_grad.name = 'sum_square_grad_' + param.name

            # Accumulate gradient
            new_sum_squared_grad = (
                sum_square_grad + T.sqr(grads[param])
            )

            # Compute update
            epsilon = lr_scalers.get(param, 1.) * learning_rate
            delta_x_t = (- epsilon / T.sqrt(new_sum_squared_grad)
                         * grads[param])

            # Apply update
            updates[sum_square_grad] = new_sum_squared_grad
            updates[param] = param + delta_x_t

        return updates

class RPROP(LearningRule):
    def __init__(
        self,
        decrease_rate=0.5,
        increase_rate=1.2,
        min_rate=1e-6,
        max_rate=50
    ):
        self.decrease_rate = sharedX(decrease_rate, 'decrease_rate')
        self.increase_rate = sharedX(increase_rate, 'increase_rate')
        self.min_rate = min_rate
        self.max_rate = max_rate

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        monitor.add_channel(
            'rprop_decrease_rate',
            ipt=None,
            val=self.decrease_rate,
            dataset=monitoring_dataset,
            data_specs=(NullSpace(), '')
        )
        monitor.add_channel(
            'rprop_increase_rate',
            ipt=None,
            val=self.increase_rate,
            dataset=monitoring_dataset,
            data_specs=(NullSpace(), '')
        )

    def get_updates(self, learning_rate, grads, lr_scalers=None, global_error=None):
        updates = OrderedDict()

        for param, grad in grads.iteritems():
            # Created required shared variables
            lr = lr_scalers.get(param, learning_rate.get_value())
            delta = sharedX(
                np.zeros_like(param.get_value()) + lr,
                borrow=True
            )
            previous_grad = sharedX(
                np.zeros_like(param.get_value()),
                borrow=True
            )

            # Name variables according to the parameter name
            if param.name is not None:
                delta.name = 'delta_'+param.name
                previous_grad.name = 'previous_grad_'+param.name

            temp = grad*previous_grad
            delta_inc = T.clip(
                T.switch(
                    T.eq(temp, 0.),
                    delta,
                    T.switch(
                        T.lt(temp, 0.),
                        delta*self.decrease_rate,
                        delta*self.increase_rate
                    )
                ),
                self.min_rate,
                self.max_rate
            )

            previous_grad_inc = T.switch(
                T.gt(temp, 0.),
                grad,
                T.zeros_like(grad)
            )

            # Calculate updates of parameters
            updated_inc = -delta*T.sgn(grad)

            # Compile the updates
            updates[param] = param + updated_inc
            updates[delta] = delta_inc
            updates[previous_grad] = previous_grad_inc

        return updates

class ARPROP(LearningRule):
    def __init__(
        self,
        decrease_rate=0.5,
        increase_rate=1.2,
        min_rate=1e-6,
        max_rate=50
    ):
        self.decrease_rate = sharedX(decrease_rate, 'decrease_rate')
        self.increase_rate = sharedX(increase_rate, 'increase_rate')
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.q = sharedX(1,'q')
        self.prev_global_error = sharedX(0,'prev_global_error')
        self.global_error = sharedX(0,'global_error')

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        monitor.add_channel(
            'rprop_decrease_rate',
            ipt=None,
            val=self.decrease_rate,
            dataset=monitoring_dataset,
            data_specs=(NullSpace(), '')
        )
        monitor.add_channel(
            'rprop_increase_rate',
            ipt=None,
            val=self.increase_rate,
            dataset=monitoring_dataset,
            data_specs=(NullSpace(), '')
        )
        monitor.add_channel(
            'q',
            ipt=None,
            val=self.q,
            dataset=monitoring_dataset,
            data_specs=(NullSpace(), '')
        )
        monitor.add_channel(
            'prev_global_error',
            ipt=None,
            val=self.prev_global_error,
            dataset=monitoring_dataset,
            data_specs=(NullSpace(), '')
        )
        monitor.add_channel(
            'global_error',
            ipt=None,
            val=self.global_error,
            dataset=monitoring_dataset,
            data_specs=(NullSpace(), '')
        )

    def get_updates(self, learning_rate, grads, lr_scalers=None,
            global_error=None):
        if global_error is None:
            raise ValueError("global_error needs to be passed to ARPROP")
        updates = OrderedDict()

        updated_q = T.switch(
                T.lt(self.prev_global_error,self.global_error),
                T.switch(T.gt(self.prev_global_error,0),
                    T.switch(T.gt(self.global_error,0),
                        self.q + 1,
                        1),
                    1),
                1)
        for param, grad in grads.iteritems():
            # Created required shared variables
            lr = lr_scalers.get(param, learning_rate.get_value())
            delta = sharedX(
                np.zeros_like(param.get_value()) + lr,
                borrow=True
            )
            previous_delta = sharedX(
                np.zeros_like(param.get_value()) + lr,
                borrow=True
            )
            previous_grad = sharedX(
                np.zeros_like(param.get_value()),
                borrow=True
            )

            # Name variables according to the parameter name
            if param.name is not None:
                delta.name = 'delta_'+param.name
                previous_grad.name = 'previous_grad_'+param.name

            temp = grad*previous_grad
            delta_inc = T.clip(
                T.switch(
                    T.eq(temp, 0.),
                    delta,
                    T.switch(
                        T.lt(temp, 0.),
                        delta*self.decrease_rate,
                        delta*self.increase_rate
                    )
                ),
                self.min_rate,
                self.max_rate
            )

            previous_grad_inc = T.switch(
                T.gt(temp, 0.),
                grad,
                T.zeros_like(grad)
            )

            # Calculate updates of parameters
            updated_inc = T.switch(
                    T.lt(self.prev_global_error,self.global_error),
                    previous_delta/T.pow(2.,updated_q),
                    -delta*T.sgn(grad))

            updated_previous_delta = T.switch(
                    T.lt(self.prev_global_error,self.global_error),
                    previous_delta, delta_inc)

            # Compile the updates
            updates[param] = param + updated_inc
            updates[delta] = delta_inc
            updates[previous_grad] = previous_grad_inc
            updates[previous_delta] = updated_previous_delta

    
#        new_prev_global_error = T.switch(T.le(self.prev_global_error,0),self.global_error,
#                        T.switch(T.lt(self.prev_global_error,self.global_error),
#                        self.prev_global_error,
#                        self.global_error))
        new_prev_global_error = self.global_error
        updates[self.q] = updated_q
        updates[self.prev_global_error] = new_prev_global_error
        updates[self.global_error] = global_error
        return updates

class RMSProp(LearningRule):
    """
    Implements the RMSProp learning rule.

    The RMSProp learning rule is described by Hinton in `lecture 6
    <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`
    of the Coursera Neural Networks for Machine Learning course.

    In short, Hinton suggests "[the] magnitude of the gradient can be very
    different for different weights and can change during learning.  This
    makes it hard to choose a global learning rate." RMSProp solves this
    problem by "[dividing] the learning rate for a weight by a running
    average of the magnitudes of recent gradients for that weight."


    Parameters
    ----------
    decay : float, optional
        Decay constant similar to that used in AdaDelta and Momentum methods.
    max_scaling: float, optional
        Restrict the RMSProp gradient scaling coefficient to values
        below `max_scaling`.

    Notes
    -----
    An instance of this LearningRule should only be used with one
    TrainingAlgorithm, and its get_updates method should be called
    only once. This is required in order to make the monitoring
    channels correctly report the moving averages.
    """

    def __init__(self, decay=0.9, max_scaling=1e5):
        assert 0. <= decay < 1.
        assert max_scaling > 0
        self.decay = sharedX(decay, 'decay')
        self.epsilon = 1. / max_scaling
        self.mean_square_grads = OrderedDict()

    @wraps(LearningRule.add_channels_to_monitor)
    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        """
        The channels added are the min, mean, and max of the
        mean_square_grad of each parameter.
        """

        channel_mapping = {
            '_min': T.min,
            '_max': T.max,
            '_mean': T.mean
        }

        for mean_square_grad in self.mean_square_grads.values():
            for suffix, op in channel_mapping.items():
                monitor.add_channel(
                    name=(mean_square_grad.name + suffix),
                    ipt=None,
                    val=op(mean_square_grad),
                    data_specs=(NullSpace(), ''),
                    dataset=monitoring_dataset)
        return

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        Provides the symbolic (theano) description of the updates needed to
        perform this learning rule. See Notes for side-effects.

        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.

        Returns
        -------
        updates : OrderdDict
            A dictionary mapping from the old model parameters, to their new
            values after a single iteration of the learning rule.

        Notes
        -----
        This method has the side effect of storing the moving average
        of the square gradient in `self.mean_square_grads`. This is
        necessary in order for the monitoring channels to be able
        to track the value of these moving averages.
        Therefore, this method should only get called once for each
        instance of RMSProp.
        """

        updates = OrderedDict()
        for param in grads:

            # mean_squared_grad := E[g^2]_{t-1}
            mean_square_grad = sharedX(param.get_value() * 0.)

            if param.name is None:
                raise ValueError("Model parameters must be named.")
            mean_square_grad.name = 'mean_square_grad_' + param.name

            if param.name in self.mean_square_grads:
                warnings.warn("Calling get_updates more than once on the "
                              "gradients of `%s` may make monitored values "
                              "incorrect." % param.name)
            # Store variable in self.mean_square_grads for monitoring.
            self.mean_square_grads[param.name] = mean_square_grad

            # Accumulate gradient
            new_mean_squared_grad = (self.decay * mean_square_grad +
                                     (1 - self.decay) * T.sqr(grads[param]))

            # Compute update
            scaled_lr = lr_scalers.get(param, 1.) * learning_rate
            rms_grad_t = T.sqrt(new_mean_squared_grad)
            rms_grad_t = T.maximum(rms_grad_t, self.epsilon)
            delta_x_t = - scaled_lr * grads[param] / rms_grad_t

            # Apply update
            updates[mean_square_grad] = new_mean_squared_grad
            updates[param] = param + delta_x_t

        return updates

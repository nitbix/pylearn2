"""
Functionality for training with NormAN.
"""
__authors__ = 'Alan Mosca'
__copyright__ = "Copyright 2014, Birkbeck, University of London"

from pylearn2.costs.cost import DefaultDataSpecsMixin, Cost

class NormAN(DefaultDataSpecsMixin, Cost):
    """
    Implements a NORmal Multiplied Attenuation Network (NormAN). This is a
    new training method which introduces a gaussian perturbation of the 
    learning rates.
    This technique also allows for a dropout rate, as described in
    "Improving neural networks by preventing co-adaptation of feature
    detectors"
    Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever,
    Ruslan R. Salakhutdinov
    arXiv 2012

    On top of the selection to include or exclude an input to a layer, we perturb
    independently at each training iteration the learning rates on each node.
    We use a truncated normal between (0,2) with mean = 1.0, and a given st dev
    parameter. No re-normalisation of the weights is required, because the expected
    mean of the weights will be the same (we are on average multiplying by 1).

    Parameters
    ----------
    default_input_include_prob : float
        The probability of including a layer's input, unless that layer appears
        in `input_include_probs`
    input_include_probs : dict
        A dictionary mapping string layer names to float include probability
        values. Overrides `default_input_include_prob` for individual layers.
    default_perturb_std : float
        The st dev of the perturbation distribution, unless that layer appears
        in `input_include_stds`
    input_perturb_stds : dict
        A dictionaty mapping string layer names to float perturbation st dev
        values. Overrides `default_perturb_std` for individual layers.
    default_input_scale : float
        During training, each layer's input is multiplied by this amount to
        compensate for fewer of the input units being present. Can be
        overridden by `input_scales`.
    input_scales : dict
        A dictionary mapping string layer names to float values to scale that
        layer's input by. Overrides `default_input_scale` for individual
        layers.
    per_example : bool
        If True, chooses separate units to drop for each example. If False,
        applies the same dropout mask to the entire minibatch.
    """

    supervised = True

    def __init__(self, default_input_include_prob=.5, input_include_probs=None,
            default_perturb_std=.1, input_perturb_stds=None,
            default_input_scale=2., input_scales=None, per_example=True):

        if input_include_probs is None:
            input_include_probs = {}
        
        if input_perturb_stds is None:
            input_perturb_stds = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        """
        .. todo::

            WRITEME
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data
        Y_hat = model.dropout_fprop(
            X,
            default_input_include_prob=self.default_input_include_prob,
            input_include_probs=self.input_include_probs,
            default_input_scale=self.default_input_scale,
            input_scales=self.input_scales,
            per_example=self.per_example
        )
        return model.cost(Y, Y_hat)

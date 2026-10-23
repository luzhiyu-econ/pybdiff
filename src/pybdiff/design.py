"""Rebuild the main model matrix with PyFixest's original formula context."""

from collections.abc import Mapping

import numpy as np
from pyfixest.estimation.formula.factor_interaction import factor_interaction
from pyfixest.estimation.formula.utils import log


def main_matrix(model):
    context = {"log": log, "i": factor_interaction, "np": np}
    if isinstance(getattr(model, "_context", None), Mapping):
        context.update(model._context)
    return model._model_spec["second_stage"].get_model_matrix(model._data, context=context)

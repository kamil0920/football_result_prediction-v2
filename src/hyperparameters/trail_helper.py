from typing import Any, Dict, Sequence

import pandas as pd


def trial2df(trial: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """
        Convert a Trial object (sequence of trial dictionaries)
        to a Pandas DataFrame.

    Parameters
    ----------
    trial : List[Dict[str, Any]]
        A list of trial dictionaries.
    Returns
    -------
    pd.DataFrame
        A DataFrame with columns for the loss, trial id, and
        values from each trial dictionary.
    """
    vals = []
    for t in trial:
        result = t['result']
        misc = t['misc']
        val = {k: (v[0] if isinstance(v, list) else v)
               for k, v in misc['vals'].items()
               }

        val['loss'] = result['loss']
        val['tid'] = t['tid']
        vals.append(val)
    return pd.DataFrame(vals)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt import space_eval


def extract_trial_results(trials, param_names):
    """
    Extracts hyperparameter values and loss from each trial into a DataFrame.

    Parameters:
      trials (hyperopt.Trials): Trials object from Hyperopt.
      param_names (list): List of parameter names to extract.

    Returns:
      DataFrame: Contains one row per trial with the parameter values and loss.
    """
    rows = []
    for trial in trials.trials:
        row = {}
        for param in param_names:
            # Each parameter's value is stored in a list in trial['misc']['vals']
            if param in trial['misc']['vals']:
                # Convert to a single value (hp.quniform returns a float)
                row[param] = trial['misc']['vals'][param][0]
            else:
                row[param] = None
        row['loss'] = trial['result']['loss']
        rows.append(row)
    return pd.DataFrame(rows)
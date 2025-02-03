import numpy as np
import pandas as pd


def jitter(df: pd.DataFrame, col: str, amount: float = 1) -> pd.Series:
    """
    Add random noise to the values in a Pandas DataFrame column.
    This function adds random noise to the values in a specified
    column of a Pandas DataFrame. The noise is uniform random
    noise with a range of `amount` centered around zero. The
    function returns a Pandas Series with the jittered values.
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    col : str
        The name of the column to jitter.
    amount : float, optional
        The range of the noise to add. The default value is 1.
    Returns
    -------
    pd.Series
        A Pandas Series with the jittered values.
    """

    vals = np.random.uniform(low=-amount / 2, high=amount / 2,
                             size=df.shape[0])
    return df[col] + vals


import plotly.graph_objects as go


def plot_3d_mesh(df: pd.DataFrame, x_col: str, y_col: str,
                 z_col: str) -> go.Figure:
    """
    Create a 3D mesh plot using Plotly.
    This function creates a 3D mesh plot using Plotly, with
    the `x_col`, `y_col`, and `z_col` columns of the `df`
    DataFrame as the x, y, and z values, respectively. The
    plot has a title and axis labels that match the column
    names, and the intensity of the mesh is proportional
    to the values in the `z_col` column. The function returns
    a Plotly Figure object that can be displayed or saved as
    desired.
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to plot.
    x_col : str
        The name of the column to use as the x values.
    y_col : str
        The name of the column to use as the y values.
    z_col : str
        The name of the column to use as the z values.
    Returns
    -------
    go.Figure
        A Plotly Figure object with the 3D mesh plot.
    """
    fig = go.Figure(data=[go.Mesh3d(x=df[x_col], y=df[y_col], z=df[z_col],
                                    intensity=df[z_col] / df[z_col].min(),
                                    hovertemplate=f"{z_col}: %{{z}}<br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>")],
                    )
    fig.update_layout(
        title=dict(text=f'{y_col} vs {x_col}'),
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col),
        width=700,
        margin=dict(r=20, b=10, l=10, t=50)
    )
    return fig

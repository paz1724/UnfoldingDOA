import numpy as np, torch
import plotly.graph_objects as go

def debug_plot(data, axes=None):
    """
    Plot 1D or multi‐curve data (mode='curves') or 2D heatmap (mode='heatmap').
    `axes` may contain:
      - mode:      'curves' or 'heatmap' (default auto)
      - x:         x‐axis array
      - y:         y‐axis array for heatmap
      - names:     list of names for each curve (only in 'curves' mode)
      - title, x_label, x_unit, y_label, y_unit
      - vlines_1, hlines_1 for color_1 dashed lines
      - vlines_2, hlines_2 for color_2 dashed lines
    """
    
    arr = data.detach().cpu().numpy() if torch.is_tensor(data) else np.asarray(data)
    axes = axes or {}

    color_1 = axes.get('color_1', 'green')
    color_2 = axes.get('color_2', 'red')

    mode = axes.get('mode')
    # auto‐detect if not specified:
    if mode is None:
        mode = 'heatmap' if arr.ndim == 2 else 'curves'

    fig = go.Figure()

    if mode == 'curves':
        # ensure shape (n_curves, N)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        n_curves, N = arr.shape
        xs = axes.get('x', np.arange(N))
        names = axes.get('names', [f'curve_{i}' for i in range(n_curves)])
        for i in range(n_curves):
            fig.add_trace(go.Scatter(
                x=xs, y=arr[i],
                mode='lines+markers',
                name=names[i]
            ))
        # dashed lines (1/2)
        for col, color in (('vlines_1',color_1),('vlines_2',color_2)):
            for xv in axes.get(col, []):
                fig.add_shape(type="line",
                              x0=xv, x1=xv,
                              y0=min(arr.min(),0),  # stretch
                              y1=arr.max(),
                              line=dict(color=color, dash='dash'))
        for row, color in (('hlines_1',color_1),('hlines_2',color_2)):
            for yv in axes.get(row, []):
                fig.add_shape(type="line",
                              x0=xs[0], x1=xs[-1],
                              y0=yv,    y1=yv,
                              line=dict(color=color, dash='dash'))

    elif mode == 'heatmap':
        xs = axes.get('x', np.arange(arr.shape[1]))
        ys = axes.get('y', np.arange(arr.shape[0]))
        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=arr,
            coloraxis="coloraxis"
        ))
        fig.update_layout(coloraxis=dict(colorscale="Viridis"))
        # add dashed lines as shapes
        for col, color in (('vlines_1',color_1),('vlines_2',color_2)):
            for xv in axes.get(col, []):
                fig.add_shape(type="line", x0=xv, x1=xv,
                              y0=ys[0], y1=ys[-1],
                              line=dict(color=color, dash='dash'))
        for row, color in (('hlines_1',color_1),('hlines_2',color_2)):
            for yv in axes.get(row, []):
                fig.add_shape(type="line", x0=xs[0], x1=xs[-1],
                              y0=yv,    y1=yv,
                              line=dict(color=color, dash='dash'))
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # axis titles with units
    xl, xu = axes.get('x_label','X'), axes.get('x_unit','')
    yl, yu = axes.get('y_label','Y'), axes.get('y_unit','')
    x_title = f"{xl} ({xu})" if xu else xl
    y_title = f"{yl} ({yu})" if yu else yl

    if axes.get('title') is not None:
        fig.update_layout(title=axes['title'])
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)

    fig.show()

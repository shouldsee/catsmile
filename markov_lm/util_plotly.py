
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
def plotly_heatmap_tracks(tz, sel=None, ZMIN= None,ZMAX =None,YMAX=0, title=''):

    z = np.stack([zz for tt,zz in tz], 1)
    BB   = z.shape[0]*z.shape[1]
    if tz[0][0] is None:
        text = (z*0).astype(str)
        text[:] = ''
        # text = np.array([None] * BB)
    else:
        text = np.stack([tt for tt,zz in tz], 1)
    # z    = np.stack([z0,z1,z2,z3,z4], 1)
    # text = np.stack([t0,t1,t2,t3,t4], 1)
    # sel = None
    if sel is not None:
        z    = z[sel]
        text = text[sel]

    BB   = z.shape[0]*z.shape[1]
    z    = z.reshape((BB,-1))
    text = text.reshape((BB,-1))
    if ZMIN is not None or ZMAX is not None:
        z = z.clip(ZMIN,ZMAX)


    if YMAX>=0:
        z = z[:YMAX]
        text = text[:YMAX]

    # text = np.tile(text[:,None],(1,3,1)).reshape((3*BB,-1))
    # z  =[sel]
    # z = z[sel]
    # text = text[sel]

    # key

    if 1:
        fig = go.Figure(data=go.Heatmap(
                            z=z,
                            text=text,
                            xgap=1,
                            ygap=1,
                            zmin=ZMIN,
                            zmax=ZMAX,
                            texttemplate="%{text}",
                            colorscale=px.colors.sequential.Blues,
                            textfont={"size":15}))
        # title = f'{key}, len:{len(x)}'
        fig['layout'].update(title=title)
    return fig

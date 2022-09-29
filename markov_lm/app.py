'''
Ref: https://www.starlette.io/requests/
'''
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.responses import HTMLResponse,Response
from starlette.templating import Jinja2Templates
import uvicorn

templates = Jinja2Templates(directory='.')

import torch
class env(object):
    conf = None
    CUDA = 0
    device = torch.device('cpu' if not CUDA else 'gpu:0')
import markov_lm.service
from markov_lm.util_html import write_png_tag
CONFS = {}
def startup():
    for k in '''
translate-german-english
fashion-mnist-compress
translate-wmt14-de2en-5k
translate-wmt14-de2en-50k
translate-wmt14-de2en-20k
translate-mutli30k-de2en-l50
translate-multi30k-de2en-l20
translate-multi30k-de2en-chardata-l100
'''.strip().splitlines():
        CONFS[k] = markov_lm.service.init_conf(k,device=env.device)
    # 'fashion-mnist-compress')
# translate-german-english
    return
# def lazy_init_conf():


app = Starlette(debug=True,on_startup = [startup])
app.mount('/static', StaticFiles(directory='../statics'), name='static')


@app.route('/')
async def homepage(request):
    template = "index.html"
    context = {"request": request}
    return templates.TemplateResponse(template, context)


@app.route('/error')
async def error(request):
    """
    An example error. Switch the `debug` setting to see either tracebacks or 500 pages.
    """
    raise RuntimeError("Oh no")


@app.exception_handler(404)
async def not_found(request, exc):
    """
    Return an HTTP 404 page.
    """
    template = "404.html"
    context = {"request": request}
    return templates.TemplateResponse(template, context, status_code=404)


@app.exception_handler(500)
async def server_error(request, exc):
    """
    Return an HTTP 500 page.
    """
    template = "500.html"
    context = {"request": request}
    return templates.TemplateResponse(template, context, status_code=500)

import io
from pprint import pprint
# @app.route('/fashion_mnist_rec',methods=["GET", "POST"])
def prepare_target(default_conf_task, plotter):
    '''
    args: plotter  a function with signature `plotter(model, images)`
    '''
    async def handler(request):
        template = "index.html"
        context = {"request": request}
        form = await  request.form()
        def _fill(globkey='',model_name='',sql='',params='',**kw):
            buf = io.StringIO()
            model = markov_lm.service.get_model(model_name, device=env.device, strict='--nostrict' not in params)
            # conf_name =
            conf_task = model.meta.get('conf_task', default_conf_task)
            conf = CONFS[conf_task]
            train_fig = plotter(model, buf = buf, dataset=conf.dataset,**conf.train_data)
            test_fig = plotter(model, buf = buf, dataset=conf.dataset,**conf.test_data)
            # fig = plotter(model, env.conf.test_data['images'], env.conf.test_data['labels'])
            x = f'''
            {model_name}
            {write_png_tag(test_fig) if test_fig is not None else ''}
            {write_png_tag(train_fig) if train_fig is not None else ''}
            '''
            buf.write(x)
            buf.seek(0)
            return buf.read()
        resp = _fill(**form)
        return HTMLResponse(resp)
    return handler

app.route('/fashion_mnist_rec',methods=["GET", "POST"])(prepare_target('fashion-mnist-compress',     markov_lm.service.plot_fashion_mnist_recon))
app.route('/fashion_mnist_perp',methods=["GET", "POST"])(prepare_target('fashion-mnist-compress',    markov_lm.service.plot_fashion_mnist_perp))
# app.route('/translation_attention',methods=["GET", "POST"])(prepare_target('translate-german-english', markov_lm.service.plot_translation_attention))
# app.route('/translation_attention',methods=["GET", "POST"])(prepare_target('translate-wmt14-de2en-20k', markov_lm.service.plot_translation_attention))
# app.route('/translation_attention',methods=["GET", "POST"])(prepare_target('translate-wmt14-de2en-5k', markov_lm.service.plot_translation_attention))
# app.route('/translation_attention',methods=["GET", "POST"])(prepare_target('translate-mutli30k-de2en-l50', markov_lm.service.plot_translation_attention))
app.route('/translation_attention',methods=["GET", "POST"])(prepare_target('translate-multi30k-de2en-l20', markov_lm.service.plot_translation_attention))
app.route('/plot_latent',methods=["GET", "POST"])(prepare_target('translate-multi30k-de2en-chardata-l100', markov_lm.service.plot_latent))


BUTTONS_HTML = ''
#
BUTTONS_HTML += '''<button onclick="submitTarget(`main-form`,`/fashion_mnist_rec`)">GO /fashion_mnist_rec</button>'''
BUTTONS_HTML +=  '''<button onclick="submitTarget(`main-form`,`/fashion_mnist_perp`)">GO /fashion_mnist_perp</button>'''
BUTTONS_HTML +=  '''<button onclick="submitTarget(`main-form`,`/translation_attention`)">GO /translation_attention</button>'''
BUTTONS_HTML +=  '''<button onclick="submitTarget(`main-form`,`/plot_latent`)">GO /plot_latent</button>'''
# SUBMIT_FASHION_MNIST_REC  = '''<button onclick="submitTarget(`main-form`,`/fashion_mnist_rec`)">GO /fashion_mnist_rec</button>'''
# SUBMIT_FASHION_MNIST_PERP = '''<button onclick="submitTarget(`main-form`,`/fashion_mnist_perp`)">GO /fashion_mnist_perp</button>'''
# SUBMIT_TRANSLATION_ATTENTION = '''<button onclick="submitTarget(`main-form`,`/translation_attention`)">GO /translation_attention</button>'''


import glob
from pandasql import sqldf
import os
import pandas as pd
@app.route('/main',methods=["GET", "POST"])
async def homepage(request):
    # template = "index.html"
    context = {"request": request}

    s = io.StringIO()
    pprint(vars(request),s)
    s.seek(0)
    form = await request.form()
    def _fill(globkey='',model_name='',sql='',**kw):
        fs = glob.glob(globkey)
        df = pd.DataFrame(
        [x.rsplit('.',1)[0].rsplit('_',2)
        + [f'<button onclick="copyText(`{x}`)">CopyModel</button> {BUTTONS_HTML}'] for x in fs],
        columns = 'model_desc epoch loss copy'.split(),
        # columns = 'dirname basename split'.split()
        # dtype= dict(loss='float')
        )
        df['loss'] = df['loss'].astype(float)
        if sql:
            df = sqldf(sql,dict(df=df))
        if 1:
            tab = df.to_html(escape=False)
        else:
            # form['keyword'])
            # fs = []
            tab = ''.join([ f'''
            <tr>
                <td>
                {x}
                </td>
                <td>
                </td>
            </tr>
            ''' for x in fs])
            tab = f'''
            {fs[:1]}
            <table border=1>
            <tbody>{tab}</tbody></table>
            '''

        SCRIPT = '''
            <script>
                function submitTarget(eid,target) {
    //                form = x.form
                    form = document.getElementById(eid);
                    console.log(form.action)
                    form.setAttribute('action', target)
                    console.log(form.action)
                    form.submit()
                }

function copyText(x){
navigator.clipboard.writeText(x);
document.getElementById("main-form")["model_name"].value = x
}

            </script>
        '''
        TEMP_MAIN = (f'''

    <html>
    <head>
        {SCRIPT}
    </head>
    <body>
    <form id='main-form' method='post' action='/'>

            <table border=1>
            <tbody>
                <tr>
                    <td>
                    <label>Form</label>
                    </td>
                    <td>
                    </td>
                </tr>

                <tr>
                    <td>
                    <label>globkey</label>
                    </td>
                    <td>
                    <input id='globkey' name='globkey' value='{globkey}' size=100 maxlength="1000"></input>
                    </td>
                </tr>

                <tr>
                    <td>
                    <label>sql</label>
                    </td>
                    <td>
                    <input id='sql' name='sql' value='{sql}' size=100 maxlength="1000" ></input>
                    </td>
                </tr>

                <tr>
                    <td>
                    <label>model_name</label>
                    </td>
                    <td>
                    <input id='model_name' name='model_name' value='model_name' size=100></input>
                    </td>
                </tr>
                <tr>
                    <td>
                    <label>params</label>
                    </td>
                    <td>
                    <input id='params' name='params' value='params' size=100></input>
                    </td>
                </tr>
            </tbody>
            </table>
            <button onclick="submitTarget('main-form','/main')">GO MAIN</button>
        </form>
        <h4>Table Result</h4>
        {tab}
        <h4>Request</h4>
        <pre>{s.read()}</pre>
        <h4>Form</h4>
        <pre>{form}</pre>
    </body>
    </html>
    ''')
        # content = '%s %s <pre>%s</pre>' % (request.method, request.url.path, s.read())


        return TEMP_MAIN
    resp = _fill(**form)
    return HTMLResponse(resp)
    # return _fill(())

@app.route('/test')
async def _app(scope, receive, send):
    assert scope['type'] == 'http'
    response = FileResponse('statics/favicon.ico')
    await response(scope, receive, send)

# if __name__ == "__main__":
#     uvicorn.run(app, host='0.0.0.0', port=9001, log_level="info", )
    # uvicorn.run(app, host='0.0.0.0', port=9001, log_level="info", reload=True)

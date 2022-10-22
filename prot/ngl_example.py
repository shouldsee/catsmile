from prot.depend import ctl; ctl.build()
#loadFile = "rcsb://1crn"
loadFile = 'https://files.rcsb.org/download/1PGB.pdb'

HEADER_JS = ''
HEADER_JS += f'''
loadFile = {loadFile!r}
'''

HEADER_JS += '''

var stage;
document.addEventListener("DOMContentLoaded", function (){
    var stage = new NGL.Stage( "viewport" );

    window.addEventListener( "resize", function( event ){
        stage.handleResize();
    }, false );

    stage.loadFile( loadFile, { defaultRepresentation: true } );
})
'''

HEADER_CSS = '''
* { margin: 0; padding: 0; }
html, body { width: 100%; height: 100%; overflow: hidden; }
'''

#'<script src="./node_modules/ngl/dist/ngl.js"></script>'
#PATH_NGL = './node_modules/ngl/dist'
buf = f'''
<html>
<head>
<script src="{ctl['init_ngl'].check_ctx}"></script>
<script>
{HEADER_JS}
</script>

<style>
{HEADER_CSS}
</style>
</head>
<body>
<div id="viewport" style="width:100%; height:100%;"></div>
</body>
</html>
'''

with open(__file__+'.html', 'w') as f:
    f.write(buf)


#### http server module

__version__ = 'v0.1'
from t1112_restore_bak import get_engine
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse as urlparse
import sys,os
import traceback
import pymssql
CONTACT_NUMBER = '400-819-9900'
##### GET FILE PATH
if 1:
	if getattr(sys, 'frozen', False):
		EXE_PATH = sys.executable
	elif __file__:
		EXE_PATH = __file__
	EXE_PATH = os.path.realpath(EXE_PATH)
	__DIR__ = os.path.dirname(EXE_PATH)

############### LOGGING
import logging
class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def main_rediect_log():
	log = logging.getLogger('foobar')
	sys.stdout = StreamToLogger(log,logging.INFO)
	sys.stderr = StreamToLogger(log,logging.ERROR)
	logging.basicConfig(
	        level=logging.DEBUG,
	        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
	        filename=EXE_PATH+'.all.log',
	        filemode='a'
	        )


def get_config(path,):
	res = glob(path+'*yaml')
	assert len(res)==1,[print(res),'没有唯一的配置文件']
	configFile = res[0]
#	configFile = EXE_PATH+'.yaml'
	with open(configFile,'rb') as f:
		buf = f.read()
		try:
			buf=buf.decode('gbk')
		except:
			buf = buf.decode('utf8')
		config = yaml.load(buf)
	return config
#	return config


#from urllib import parse_qs
#import jinja2
#host = ('127.0.0.1',8090)
import io
import json
import sqlite3
import pandas as pd
from datetime import datetime

from enum import IntEnum
EN2CN = '''
ScanCode 条码
ScanCodeText 条码文本
ReplSmallPackQty 补药盒数
ReplBigPackQty	补药包数
SpecificationQuantityDesc 打包数
MedicationCode 药品代码
MedicationName 药品名
MedicationSpec 药品规格
BrandName 商品名
Manufacturer   药品厂家
High	上限
RecentMaxQty	指导上限
TargetBigPackQty 指导上限
#RecentMaxQty1 近期单日最大机发包数
Quantity	库存包数
DeviceStorageLocationDesc	库位
SpecificationQuantity	每包盒数

InputType	入药模式
ReplRateBar	缺药率图
ReplRate	缺药率
Buttons	按钮
ReplTaskStatus	任务状态代码
ReplTaskStatusDesc 任务状态
StockInDay 库存可用天数
DefaultStorageLocation 货位码
'''
EN2CN = dict([x.split()[:2] for x in EN2CN.strip().splitlines() if x.strip()!='' and not x.startswith('#')])
CN2EN = {y:x for x,y in EN2CN.items()}
for k in CN2EN:
	print('  %s: 1'%k)
class ReplTaskStatus(IntEnum):
	Default = 0
	NoStock = 1
#	Packed  = 2
	Prolog  = 2
RTS = ReplTaskStatus
class ReplTaskStatusCN(IntEnum):
	未开始	  = 0
	本日缺药	= 1
#	已打包	  = 2
	已入PROLOG  = 2
RTSC = ReplTaskStatusCN
def dbChangeReplTaskStatus(db,p,status):
	p['TaskDate'] = [datetime.now().strftime('%Y-%m-%d')]
	db.execute(f'''Replace into ReplTask(MedicationSpecificationId, DeviceStorageLocationId,ReplTaskDate, Status)
	VALUES('{p["MedicationSpecificationId"][0]}','{p["DeviceStorageLocationId"][0]}','{p["TaskDate"][0]}',{int(status)})''')
	db.commit()
	return 0
# get_cache_repl_data_last = [None,None]
get_cache_repl_data_last= {}
#from datetime import datedelta
def get_cache_repl_data(conn,apiurl,SQL_LOOKBACK_DAY,DATENOW,CONFIG_TARGET,CONFIG_DEBUG):
	# last,val = get_cache_repl_data_last
	keyTuple = (SQL_LOOKBACK_DAY, DATENOW,CONFIG_TARGET)
	lastTime,val = get_cache_repl_data_last.get(keyTuple,(None,None))

	if val is None or (datetime.now() -  lastTime).seconds > 30:
		val =  get_repl_data(conn,apiurl,SQL_LOOKBACK_DAY,DATENOW,CONFIG_TARGET,CONFIG_DEBUG)
		get_cache_repl_data_last[keyTuple] = [datetime.now(),val]
		print('!'*10 + '[Recalculateing data ......]')
	else:
		print('!'*10+'[SkippedReplData]')
	return val

from pprint import pprint
template_dti = {
		'BasicUnit': '',
	  'BasicUnitQty': 1,
	  'CandicatedInventoryList': [],
	  'CandicatedInventoryListForPercolating': [],
	  'Comment': '',
	  'Instruction': 'v_Instruction',
	  'IsDismounted': False,
	  'Manufacturer': 'v_Manufacturer',
	  'MedicationCode': 'v_MedicationCode',
	  'MedicationId': 0,
	  'MedicationName': 'v_MedicationName',
	  'MedicationSpec': 'v_MedicationSpec',
	  'OrdingUnitDesc': '',
	  'OrdingUnitName': '包',
	  'PackageType': '包',
	  'Prescription': None,
	  'PrescriptionId': 0,
	  'PrescriptionItemId': 0,
	  'RequestedQty': 0,
	  'RequestedQtyByUnit': 1,
	  'SpecificationQuantity': 1}
def get_currnet_task_id(db):
					v = pd.read_sql('select Value from CommonDict where Key = "CurrentTaskId"',db)
					if not len(v):
						v = 0
					else:
						v = v.values.ravel()[0]
					return [v,'补药%09d'%v]
from datetime import timedelta

def config_web(conn,apiurl,db,CONFIG_USE_PRINT,CONFIG_SQL_LOOKBACK_DAY,CONFIG_EXPIRY_DATE, CONFIG_MAIN_PAGE_COLUMNS_ENABLER_PAIR,CONFIG_TARGET,LICENSE,CONFIG_DEBUG,
CONFIG_PRINT_INSTRUCTION_TEMPLATE,CONFIG_PRINT_PATIENT_NAME_TEMPLATE):
	class Request(BaseHTTPRequestHandler):
		timeout = 5
		server_version = 'What'
		def do_GET(self):
			parsed = urlparse.urlparse(self.path)
			print(parsed)
			if parsed.path == '/backend/ReplTaskStatusUpdate':
				self.send_response(200)
				self.send_header("Content-Type","data/json; charset=utf-8")
				self.end_headers()
				print(f'[params] {parsed.params}')
				p = urlparse.parse_qs(urlparse.unquote(parsed.query))
				dbChangeReplTaskStatus(db,p,p['Status'][0])
				self.wfile.write(json.dumps({'result':1}).encode())

			elif parsed.path == '/backend/ReplTaskStatus':
				self.send_response(200)
				self.send_header("Content-Type","text/html; charset=utf-8")
				self.end_headers()
				buf = pd.read_sql('select * from ReplTask',db).to_html()
				self.wfile.write(buf.encode())


			else:
				self.send_response(200)
				self.send_header("Content-Type","text/html; charset=utf-8")
				self.end_headers()

				self.do_POST()

#				self.wfile.write(buf.encode())
#			else:
#				return 0

		def do_POST(self):
			self.send_response(200)
			self.send_header("Content-Type","text/html; charset=utf-8")
			self.end_headers()
			try:
				parsed = urlparse.urlparse(self.path)
				length = int(self.headers['content-length'] or 0)
				fdata = self.rfile.read(length)
				print(fdata)
				print(fdata.decode('utf8'))
				if 'application/json' in (self.headers['Content-Type'] or ''):
					fs = json.loads(fdata.decode())
				else:
					fs = urlparse.parse_qs(urlparse.unquote(fdata.decode()))
				validate_license_date(LICENSE,CONFIG_EXPIRY_DATE)
	#			fs = urlparse.parse_qs(urlparse.urlparse(fdata).query)
				buf = self.do_unsafe_POST(parsed,fdata,fs)
			except Exception as e:
#				buf = str(e)
				buf = traceback.format_exception(sys.exc_info()[0],
                        sys.exc_info()[1] , sys.exc_info()[2])
				buf = ''.join(buf)
				buf = f'<pre>{buf}</pre>'
				raise e
			finally:
				self.wfile.write(buf.encode())
			return

		def do_unsafe_POST(self,parsed,fdata,fs):

			if parsed.path == '/backend/StartReplTask':
				v = get_currnet_task_id(db)
				v[0]+=1
				db.execute(f'''Replace into CommonDict(Key,Value)
				Values("CurrentTaskId",{v[0]}) ''')
				db.commit()
				par = fs['Parameters']
				if par['Success']:
					taskPatientName = CONFIG_PRINT_PATIENT_NAME_TEMPLATE.format(**par)
					taskPrescriptionId = v[1]
					return t20220215_print_repl.taskadmit_and_checkin_and_remove(apiurl,conn, taskPrescriptionId, fs['PrescriptionItemList'],taskPatientName)
				else:
					return json.dumps({'Success':False})
			else:
				pass

			fs.setdefault('DeviceStorageLocationDesc',['VMAX1','VMAX2'])
			fs.setdefault('InputType','1-打包  2-手动  3-自动 '.split())
			fs.setdefault('ReplTaskStatusDesc',[RTSC(xx)._name_ for xx in range(3)])
			fs.setdefault('ReplRate',['0'])
			fs.setdefault('StockInDay',['1'])
			fs.setdefault('printPageIndex',['1'])
			fs.setdefault('recordPerPage',['10'])
			fs.setdefault('filterStockInDayPercent',['0'])
			print(fs)
			DeviceStorageLocationDesc = fs.get('DeviceStorageLocationDesc',)
			InputType = fs.get('InputType',)
			ReplRate  = int(fs.get('ReplRate',)[0])
			StockInDay= float(fs.get('StockInDay',)[0])
			printPageIndex = int(fs['printPageIndex'][0])
			recordPerPage  = int(fs['recordPerPage'][0])
			printPrescriptionId = get_currnet_task_id(db)[1]
			filterStockInDayPercent    = int(fs['filterStockInDayPercent'][0])

			def make_button(xx):
				par = urlparse.urlencode({"MedicationSpecificationId":xx.MedicationSpecificationId,"DeviceStorageLocationId":xx.DeviceStorageLocationId,"Status":int(RTS.NoStock)})
				y = f'''
				<a href="javascript:void(0)" onclick="javascript:httpGet('/backend/ReplTaskStatusUpdate?{par}')"><button>本日缺药</button></a>
				'''.strip()
				par = urlparse.urlencode({"MedicationSpecificationId":xx.MedicationSpecificationId,"DeviceStorageLocationId":xx.DeviceStorageLocationId,"Status":int(RTS.Prolog)})
				y+=f'''
				<a href="javascript:void(0)" onclick="javascript:httpGet('/backend/ReplTaskStatusUpdate?{par}')"><button>已入槽</button></a>
				'''.strip()
				par = urlparse.urlencode({"MedicationSpecificationId":xx.MedicationSpecificationId,"DeviceStorageLocationId":xx.DeviceStorageLocationId,"Status":int(RTS.Default)})
				y+=f'''
				<a href="javascript:void(0)" onclick="javascript:httpGet('/backend/ReplTaskStatusUpdate?{par}')"><button>重置</button></a>
				'''.strip()
				return y

			DATENOW = datetime.now().strftime('%Y-%m-%d')
			# DATENOW = '2021-12-22'
			# DATENOW = '2020-09-09'
			x = replData = get_cache_repl_data(conn,apiurl, CONFIG_SQL_LOOKBACK_DAY,DATENOW,CONFIG_TARGET,CONFIG_DEBUG)
			keys = 'MedicationSpecificationId DeviceStorageLocationId'.split()
			y = pd.read_sql(f'select * from ReplTask where ReplTaskDate = "{datetime.now().strftime("%Y-%m-%d")}"',db)
			print(y)
			x = x.set_index(keys,drop=True).join(y.set_index(keys,drop=True),how='left',).reset_index()
			x0 = x.copy()
#			x = t1223_get_repl.view_table2(x)
			if 1:
				x = x['ScanCodeText BrandName ScanCode MedicationName   MedicationCode Manufacturer MedicationSpec   High RecentMaxQty Quantity DeviceStorageLocationDesc SpecificationQuantity InputType   DefaultStorageLocation'.split()]
				# x = x['ScanCodeText BrandName ScanCode MedicationName   MedicationCode Manufacturer MedicationSpec   RecentMaxQty DeviceStorageLocationDesc SpecificationQuantity InputType DefaultStorageLocation'.split()]
				x = x.sort_values('DeviceStorageLocationDesc InputType MedicationName '.split(),ascending=False)
				x = x.reset_index(drop=True)

			x['Buttons'] = [make_button(xx) for xx in x0.itertuples()]
			x['ReplTaskStatus'] = x0['Status'].fillna(ReplTaskStatus.Default)
			print(dir(ReplTaskStatusCN(0)))
			x['ReplTaskStatusDesc'] = x['ReplTaskStatus'].map(lambda x:(ReplTaskStatusCN(x)._name_))


			####
			def get_target(xx):
				return xx.RecentMaxQty
			targetInDay = pd.Series([get_target(xx) for xx in x.itertuples()]).astype(int)
			target = targetInDay*StockInDay

			x['Quantity'] = x['Quantity'].astype(int)
			x['TargetBigPackQty'] = target = target.astype(int)
#			target = x['RecentMaxQty']
			x['ReplBigPackQty'] =  target - x['Quantity']
			x['ReplSmallPackQty']= x['ReplBigPackQty'] * x['SpecificationQuantity']
			x['ReplRate'] = (x['ReplBigPackQty']/target.clip(1,None) *100).fillna(0.).astype(int)
			x['ReplRateBar'] = ['X'* int(xx/10) for xx in x['ReplRate'] ]
			x['StockInDay'] = (x['Quantity']/(targetInDay.clip(1,None)))
			x['SpecificationQuantityDesc'] = x['SpecificationQuantity'].astype(str) + '盒/包'
			x = x.sort_values('DeviceStorageLocationDesc InputType ReplBigPackQty MedicationName '.split(),ascending=False)


			x = x.loc[x.ReplTaskStatusDesc.isin(fs['ReplTaskStatusDesc'])]
			x = x.loc[x.DeviceStorageLocationDesc.str.upper().isin(fs['DeviceStorageLocationDesc'])]
			x = x.loc[x.InputType.isin(fs['InputType'])]
#			x = x.loc[x['High'] != 200]
			x = x.loc[x.StockInDay < StockInDay]
			x = x.loc[x.High!=0]
			x = x.loc[x.InputType!='4-其他']
			if filterStockInDayPercent>0:
				x = x.loc[x.StockInDay < (filterStockInDayPercent/100.)]
#			x = x.loc[x['ReplRate']>ReplRate]


			replData = x[[v for k,v in CONFIG_MAIN_PAGE_COLUMNS_ENABLER_PAIR]]
#			replData = x['ScanCode ReplSmallPackQty ReplBigPackQty SpecificationQuantityDesc MedicationName TargetBigPackQty Quantity DeviceStorageLocationDesc InputType ReplRate ReplRateBar StockInDay Buttons ReplTaskStatusDesc DefaultStorageLocation'.split()]
			replData = replData.reset_index(drop=True)
			replData.columns = [EN2CN.get(xx,xx) for xx in replData.columns]
			replData.columns.name = '序号'

			formHTML = f'<textarea>{repr(fs)}</textarea>'


			print(formHTML)
			def isChecked(fs,name,value):
				if value in fs[name]:
					return 'checked'
				else:
					return ''


			def add_checkbox(fs,name,value):
				buf = f'''<input style="width:30px;height:30px;" type="checkbox" name="{name}" id="{name}" value="{value}" {isChecked(fs,name,value)}/><label>{value}</label>'''
				return buf
			DEBUG = 1 if 'debug' in parsed.params  or 'debug' in parsed.query else 0
			def make_dti_from_rec(xx):
				# assert 0
				x = {
		'BasicUnit': '',
	  'BasicUnitQty': 1,
	  'CandicatedInventoryList': [],
	  'CandicatedInventoryListForPercolating': [],
	  'Comment': '',
	  'Instruction': CONFIG_PRINT_INSTRUCTION_TEMPLATE.format(xx=xx),
	  'IsDismounted': False,
	  'Manufacturer': xx.Manufacturer,
	  'MedicationCode': xx.MedicationCode,
	  'MedicationId': 0,
	  'MedicationName': xx.MedicationName,
	  'MedicationSpec': xx.MedicationSpec,
	  'OrdingUnitDesc': '',
	  'OrdingUnitName': '盒',
	  'PackageType': '盒',
	  'Prescription': None,
	  'PrescriptionId': 0,
	  'PrescriptionItemId': 0,
	  'RequestedQty': xx.ReplSmallPackQty,
	  'RequestedQtyByUnit': 1,
	  'SpecificationQuantity': 1}
				return x
			printDataList = [make_dti_from_rec(xx) for xx in x.itertuples()]

			# printButton = '''<a href='javascript:void(0)' onclick='buf={};(new FormData(mainForm)).forEach(function(v,k){buf[k]=v;});console.log(buf)'>打印</a>'''
			# printButton = '''<a href='javascript:void(0)' onclick='dx=mainForm.recordPerPage.value; x=mainForm.printPageIndex.value; dat = printData.slice((x-1)*dx,x*dx);httpPost(`/backend/StartReplTask`,JSON.stringify( {HisPrescriptionId:mainForm.printPrescriptionId.value,PrescriptionItemList:dat}) )'>打印</a> '''
			### add print button
			printButton = '''
			<a href='javascript:void(0)' onclick='dx=mainForm.recordPerPage.value; x=mainForm.printPageIndex.value; dat = printData.slice((x-1)*dx,x*dx);httpPost(`/backend/StartReplTask`,JSON.stringify( {HisPrescriptionId:`%s`, PrescriptionItemList:dat,Parameters:formToJson(document.getElementById(`mainForm`))}) )'><button>打印</button></a>'''%datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
			formHTML = f'<h2>补药小程序 {__version__} 有效期:{CONFIG_EXPIRY_DATE}</h2>'
			if (datetime.now() + timedelta(days=31)).isoformat() > CONFIG_EXPIRY_DATE:
				formHTML+=f'<h3 style="color:blue">程序即将过期，请及时联系{CONTACT_NUMBER}更新许可证,当前MDIS为{apiurl}</h3>'

			printSection = ''
			if CONFIG_USE_PRINT:
				printSection += f'''
			<label>打印单号:{printPrescriptionId}</label>
		<!--	<input name='printPrescriptionId' value={printPrescriptionId}></input>  -->

			<label>每页打印条数</label><input name='recordPerPage' value={recordPerPage} size=2></input>
			<label>打印页数</label><input name='printPageIndex' value={printPageIndex} size=2></input><label>/{x.shape[0]//recordPerPage+1}</label>
			{printButton}
			'''

			if DEBUG:
				formHTML += f'''
			{repr(fs) if DEBUG else ''}
<a href='javascript:void(0)' onclick='javascript:httpPost("/",`{{"hi":1}}`);'>testPost</a>
			'''
			def get_target_msg(CONFIG_TARGET):
				if CONFIG_TARGET   == 'MaxByMachineUsage':
					msg = '指导上限=库存可用天数 x (统计区间内对应机器，对应品规，峰值日出药量)'
				elif CONFIG_TARGET == 'MaxByTotalUsage':
					msg = '指导上限=库存可用天数 x (统计区间内药房发药，对应品规，峰值日出药量)'
				elif CONFIG_TARGET =='AvgByTotalUsage':
					msg = '指导上限=库存可用天数 x (统计区间内药房发药，对应品规，日均出药量)'
				else:
					msg = 'CONFIG_TARGET = {CONFIG_TARGET}'
				return msg


			tabLeft = f'''

			<form action="" method="post" id="mainForm">
			<h4>
			<input type="checkbox" style="width:30px;height:30px;"name="DeviceStorageLocationDesc" id="DeviceStorageLocationDesc" value="VMAX1" {isChecked(fs,"DeviceStorageLocationDesc","VMAX1")} /><label>VMAX1</label>
			<input type="checkbox" style="width:30px;height:30px;" name="DeviceStorageLocationDesc" id="DeviceStorageLocationDesc" value="VMAX2" {isChecked(fs,"DeviceStorageLocationDesc","VMAX2")} /><label>VMAX2</label>
			<br/>
			<br/>
			<input type="checkbox" style="width:30px;height:30px;"name="InputType" id="InputType" value="1-打包" {isChecked(fs,"InputType","1-打包")}/><label>1-打包</label>
			<input type="checkbox" style="width:30px;height:30px;"name="InputType" id="InputType" value="2-手动" {isChecked(fs,"InputType","2-手动")}/><label>2-手动</label>
			<input type="checkbox" style="width:30px;height:30px;" name="InputType" id="InputType" value="3-自动" {isChecked(fs,"InputType","3-自动")}/><label>3-自动</label>
			<br/>
			<br/>
			{add_checkbox(fs,'ReplTaskStatusDesc',RTSC(0)._name_)}
			{add_checkbox(fs,'ReplTaskStatusDesc',RTSC(1)._name_)}
			{add_checkbox(fs,'ReplTaskStatusDesc',RTSC(2)._name_)}

			<br/>
<!--			<label>缺药率:</label><input name='ReplRate' value={ReplRate}></input> -->
			<label>库存可用天数</label><input name='StockInDay' value={StockInDay} size=2></input>
			<label>{get_target_msg(CONFIG_TARGET) }</label>
			<br/>
			<label>低库存筛选</label><input name='filterStockInDayPercent' value={filterStockInDayPercent} size=2></input>% 若不为0,则筛选低于此百分比的紧急缺药品种
			<br/>
			<input type='submit' value="查询" method='post'></input>
			<br/>
			</h4>

			{printSection}
</form>
'''

			tdf = pd.read_csv(io.StringIO(f'''
条目数量:{replData.shape[0]}
总盒数:{(x.ReplBigPackQty*x.SpecificationQuantity).sum()}
手工补药盒数:{(x["ReplBigPackQty"][x["InputType"].isin('2-手动'.split())]*x.SpecificationQuantity).sum()}
手工补药包数:{x["ReplBigPackQty"][x["InputType"].isin('2-手动'.split())].sum()}
手工补药品规:{x["ReplBigPackQty"][x["InputType"].isin('2-手动'.split())].__len__()}
自动补药盒数:{(x["ReplBigPackQty"][x["InputType"].isin('3-自动'.split())]*x.SpecificationQuantity).sum()}
自动补药包数:{x["ReplBigPackQty"][x["InputType"].isin('3-自动'.split())].sum()}
自动补药品规:{x["ReplBigPackQty"][x["InputType"].isin('3-自动'.split())].__len__()}
打包盒数:{(x["ReplBigPackQty"][x["InputType"].isin('1-打包'.split())]*x.SpecificationQuantity).sum()}
打包包数:{x["ReplBigPackQty"][x["InputType"].isin('1-打包'.split())].sum()}
打包品规:{x["ReplBigPackQty"][x["InputType"].isin('1-打包'.split())].__len__()}
'''.strip()),sep=':',header=None)
			tabRight = tdf.to_html(header=None,index=0)
			formHTML += f'''

			<table border="1" class="dataframe">
		  <thead>
			<tr style="text-align: left;">
			  <th>{tabLeft}</th>
			  <th>{tabRight}</th>
			</tr>
		  </thead>
		  <tbody>
		   </tbody>
		</table>
			'''

			html_head = '''
		<head>
		<script>

		function httpGet(url){

		var req = new XMLHttpRequest();
		req.open("GET",url,true);
		req.send(null);
		return req.responseText
}

		function httpPost(url,data,content_type='application/json;charset=UTF-8'){
		var req = new XMLHttpRequest();
		req.open("POST",url,true);
		req.setRequestHeader('Content-Type',content_type);
		req.send(data);
		return req.responseText
}
		var printData=JSON.parse(`%s`)


function formToJson(x){ obj={}; for(var i=0; i<x.elements.length;i++){
xx = x.elements[i];
if(obj[xx.name]==null){obj[xx.name]=[]};
if(xx.checked){
 obj[xx.name].push(xx.value);
console.log(xx)

}
}

obj['Success']=true;
['DeviceStorageLocationDesc','InputType'].forEach(function(k){
console.log(k)
if(obj[k].length>1){
  window.alert('打印失败,打印时参数不能多选'+obj[k])
  obj['Success']=false
}
})
return obj}
		</script>
		</head>
			'''%(json.dumps(printDataList))
			#x.iloc[:,:3].to_json(orient='records'))
			fformat = '{:.2f}'.format
			buf = f'''
				<!DOCTYPE HTML>
		<html>
		{html_head}
		<body>

		{formHTML}
		{replData.to_html(float_format=fformat,escape=False)}
		</body>
		</html>
				'''
			return buf
		# self.wfile.write(buf.encode())

	#	self.wfile.write(repr(fs).encode())
		# print(fs)
	return Request

from glob import glob
import http.server
import yaml
#import simpleHTTPServer
import base64
'''
import base64
lv= '10.2.251.98_2022-09-01' ;print(lv)
ltxt = base64.b64encode(lv.encode('ascii')).decode('ascii')
print(ltxt)
val = base64.b64decode(ltxt.encode('ascii')).decode('ascii')
host_ip, expiry_date = val.split('_')
'''

#licenseText = base64.b64decode(LicenseF64.encode('ascii')).decode('ascii')

def validate_license(x,ltxt):
	val = base64.b64decode(ltxt.encode('ascii')).decode('ascii')
	lic_ip, expiry_date = val.split('_')
	assert x['MDIS_IP']== lic_ip,f'许可证验证失败:MDIS_IP为{lic_ip},应为{x["MDIS_IP"]}'
	return expiry_date

def validate_license_date(ltxt,expiry_date):
	assert datetime.now().isoformat() < expiry_date,f'许可证验证失败:已过期{expiry_date}.请联系{CONTACT_NUMBER}.当前许可证:{ltxt}'
	print(f'许可验证通过!有效期:{expiry_date}')

def validate_column_enabler(enabler, cols):
	ks = []
	for k,v in enabler.items():
		assert k in cols, '不能识别列名:%r, \n[可选列]:%s'%(k,'\n'.join(list(CN2EN)))
		if v!=1:
			continue
		ks.append((k,cols[k]))
	return ks

def main():
	dconf = dict(
	LICENSE = 'MTcyLjE4LjE4LjIyXzIwMjItMDktMDE=',
#	licenseText = '',
#	licenseExpiration = '2022-09-01',
	# SQL_IP = None,
	SQL_IP = None,
	MDIS_IP = '10.242.205.53',
	HOST_IP = None,
	SQL_PORT = '1433',
	CONFIG_USE_PRINT = 0,
	CONFIG_TARGET = 'MaxByMachineUsage',
	CONFIG_DEBUG=0,
	CONFIG_PRINT_PATIENT_NAME_TEMPLATE='''补药任务-{DeviceStorageLocationDesc[0]}-{InputType[0]}''',
	CONFIG_PRINT_INSTRUCTION_TEMPLATE='{xx.ReplBigPackQty}包 * {xx.SpecificationQuantity}盒/包\\n{xx.ScanCode}\\n{xx.DeviceStorageLocationDesc}'
	)

	x = conf = get_config(EXE_PATH)
	for k,v in dconf.items():
		conf[k] = conf.get(k,None) or v
#		conf.setdefault(k,v)
	x['SQL_IP'] = x['SQL_IP'] or x['MDIS_IP']
	x['HOST_IP']= x['HOST_IP']or x['MDIS_IP']
	x['SQL_CONN_STRING'] = 'mssql+pyodbc://PharmacyWorks:!QAZ2wsx@{SQL_IP}:{SQL_PORT}/{DB_NAME}'.format(**conf)
	x['apiurl'] = f'http://{x["MDIS_IP"]}:1252'
	x['CONFIG_EXPIRY_DATE'] = validate_license(x,x['LICENSE'])
	x['CONFIG_MAIN_PAGE_COLUMNS_ENABLER_PAIR'] = validate_column_enabler(x.pop('CONFIG_MAIN_PAGE_COLUMNS_ENABLER',None), CN2EN)
#	x['CONFIG_PRINT_INSTRUCTION_TEMPLATE']
	xp  = x.copy()
	xp.pop('SQL_CONN_STRING',None)
	pprint(xp)
	main_rediect_log()
	_main(**x)

def _main(apiurl,HOST_IP,HOST_PORT,SQL_CONN_STRING,CONFIG_USE_PRINT, CONFIG_SQL_LOOKBACK_DAY,CONFIG_EXPIRY_DATE,CONFIG_MAIN_PAGE_COLUMNS_ENABLER_PAIR,
CONFIG_TARGET,
LICENSE,
CONFIG_DEBUG,
CONFIG_PRINT_INSTRUCTION_TEMPLATE,
CONFIG_PRINT_PATIENT_NAME_TEMPLATE,
**kw):


	host = (HOST_IP,HOST_PORT)
	#x['HOST_IP'],x['HOST_PORT'])
	db = sqlite3.connect(EXE_PATH + '.sqlite')
	db.execute('''Create Table if not exists ReplTask(MedicationSpecificationId number ,DeviceStorageLocationId string, ReplTaskDate string, Status number
	, unique(MedicationSpecificationId,DeviceStorageLocationId,ReplTaskDate)
	);

	''')
	db.execute('''Create Table if not exists CommonDict( Key string, Value number, unique (Key));
	''')


	conn = get_engine(SQL_CONN_STRING)
	Request = config_web(conn,apiurl,db,
		CONFIG_PRINT_INSTRUCTION_TEMPLATE=CONFIG_PRINT_INSTRUCTION_TEMPLATE,
		CONFIG_USE_PRINT=CONFIG_USE_PRINT,
		CONFIG_SQL_LOOKBACK_DAY=CONFIG_SQL_LOOKBACK_DAY,
		CONFIG_EXPIRY_DATE=CONFIG_EXPIRY_DATE,
		CONFIG_MAIN_PAGE_COLUMNS_ENABLER_PAIR = CONFIG_MAIN_PAGE_COLUMNS_ENABLER_PAIR,
		LICENSE=LICENSE,
		CONFIG_TARGET=CONFIG_TARGET,
		CONFIG_DEBUG=CONFIG_DEBUG,
		CONFIG_PRINT_PATIENT_NAME_TEMPLATE=CONFIG_PRINT_PATIENT_NAME_TEMPLATE)

	server = HTTPServer(host,Request)
	print("Starting server at %s %s"%host)
	server.serve_forever()

if __name__ =='__main__':
		main()
#import simpleHTTPServer

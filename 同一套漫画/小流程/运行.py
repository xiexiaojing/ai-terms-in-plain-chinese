"""本机文件流水线：实际排版、独立检查、等待确认、保存交付包。"""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
import sys,json,subprocess,hashlib,shutil,zipfile,time,os,argparse
D=Path(__file__).resolve().parent;ROOT=D.parent;RUN=D/'运行记录';STATE=RUN/'当前状态.json'
def save(s):
 RUN.mkdir(exist_ok=True);STATE.write_text(json.dumps(s,ensure_ascii=False,indent=2));return s
def load():return json.loads(STATE.read_text()) if STATE.exists() else {'任务':[],'外部核对':[],'阶段':'尚未运行'}
def run(skip=False,missing=False):
 now=time.strftime('%Y%m%d-%H%M%S')+'-'+str(time.time_ns()%1000000);folder=RUN/now;folder.mkdir(parents=True)
 state={'批次':now,'阶段':'正在制作','任务':[],'外部核对':[],'本轮模型调用':0};save(state)
 for n in range(1,4):
  job={'编号':n,'计划':'制作第'+str(n)+'份','状态':'未启动'};state['任务'].append(job)
  if skip and n==2:save(state);continue
  name='小流程/运行记录/'+now+'/成品-'+str(n)
  args=[sys.executable,str(ROOT/'制作.py'),'--输出',name]
  if missing and n==2:args += ['--规格','小流程/不存在的规格.txt']
  started=time.time(); r=subprocess.run(args,capture_output=True,text=True);job.update({'状态':'已启动','启动时间':started,'制作退出码':r.returncode,'制作返回':r.stdout.strip(),'制作错误':r.stderr.strip()})
  if r.returncode:job['状态']='停止：缺材料';save(state);continue
  svg=ROOT/(name+'.svg');check=subprocess.run([sys.executable,str(D/'检查.py'),str(svg),str(ROOT/'文案.json')],capture_output=True,text=True)
  report=json.loads(check.stdout);(folder/('检查-'+str(n)+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2))
  job.update({'状态':'文字通过，待看图确认' if check.returncode==0 else '检查失败','成品':str(svg.relative_to(ROOT)),'成品SHA256':report['成品SHA256'],'检查返回':report,'检查退出码':check.returncode})
  save(state)
 state['阶段']='待确认';return save(state)
def external():
 s=load();s['外部核对']=[{'编号':j['编号'],'结果':'漏启动' if '启动时间' not in j else ('无合格交付' if j.get('检查退出码')!=0 else '实际成品与检查记录齐全')} for j in s['任务']];return save(s)
def approve(n,operator):
 s=load();j=next(j for j in s['任务'] if j['编号']==n)
 if j.get('检查退出码')!=0:raise ValueError('该任务没有通过文字检查')
 p=ROOT/j['成品'];h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h!=j['成品SHA256']:raise ValueError('检查后的成品已变化，请重新检查')
 archive=p.parent/('交付包-'+str(n)+'.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for f in [p,p.with_suffix('.html'),ROOT/'原故事.txt',ROOT/'规格.txt',ROOT/'文案.json',p.parent/('检查-'+str(n)+'.json')]:z.write(f,f.name)
 j.update({'状态':'已确认，已保存本机交付包','确认者':operator,'确认时间':time.time(),'交付包':str(archive.relative_to(ROOT)),'已对外发布':False});return save(s)
PAGE="""<!doctype html><meta charset="utf-8"><title>漫画交付 · 本机流水线</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f5f5f3;color:#242424;font:22px 'PingFang SC',sans-serif}main{width:1080px;margin:28px auto}header{display:flex;justify-content:space-between;align-items:center}h1{font-size:32px;margin:0}small{color:#656565}button{font:inherit;border:1px solid #999;background:white;border-radius:8px;padding:10px 18px;cursor:pointer}button.primary{background:#242424;color:white;border-color:#242424}.steps{display:flex;gap:16px;margin:22px 0}.step{background:white;border:1px solid #ddd;border-radius:10px;flex:1;padding:18px}.step b{display:block;margin-bottom:8px;font-size:23px}.arrow{align-self:center}article{background:white;border-radius:10px;border:1px solid #ddd;padding:20px}table{width:100%;border-collapse:collapse;margin-top:16px}td,th{text-align:left;border-bottom:1px solid #ddd;padding:14px 12px}th{color:#555;font-weight:500}aside{display:flex;align-items:center;gap:18px;margin-top:18px}.notice{background:#efefeb;padding:16px;border-radius:8px;flex:1}label{margin-left:20px}#status{font-size:20px;color:#555}a{color:#242424}footer{margin-top:16px;font-size:17px;color:#555}</style>
<main><header><h1>一套漫画，走完交付流程</h1><small>本机真实文件 · 当前模型调用 0 次</small></header>
<div class="steps"><div class="step"><b>① 启动制作</b>读取故事、规格与文案</div><span class="arrow">→</span><div class="step"><b>② 保存成品</b>生成 SVG 与可编辑网页</div><span class="arrow">→</span><div class="step"><b>③ 独立检查</b>另一个程序核对八条文字</div><span class="arrow">→</span><div class="step"><b>④ 等待确认</b>看图后保存本机交付包</div></div>
<article><button class="primary" onclick="action('/run')">运行三次制作</button><label><input type="checkbox" id="skip">跳过第二次</label><label><input type="checkbox" id="missing">第二次缺规格</label><button onclick="action('/check')" style="float:right">外部核对计划</button><table><thead><tr><th>计划任务</th><th>实际结果</th><th>独立核对</th><th>成品与确认</th></tr></thead><tbody id="rows"></tbody></table><aside><div class="notice">文字检查通过后仍停在这里。打开成品，确认人物、情节和排版，再点击“确认这一版”。</div></aside><p id="status"></p></article><footer>画面沿用已生成底图；本次流水线只排字和核验。确认仅保存本机文件，页面不会向任何平台发布。</footer></main>
<script>async function render(){let s=await(await fetch('/state')).json();document.querySelector('#rows').innerHTML=s.任务.map(j=>`<tr><td>第${j.编号}次制作</td><td>${j.状态}</td><td>${(s.外部核对.find(x=>x.编号===j.编号)||{}).结果||'尚未核对计划'}</td><td>${j.成品?`<a target="_blank" href="/files/${j.成品.replace('.svg','.html')}">打开成品</a>　${j.状态.startsWith('文字通过')?`<button onclick="action('/approve?n=${j.编号}')">确认这一版</button>`:'已确认版本留档'}`:'没有成品'}</td></tr>`).join('');document.querySelector('#status').textContent=s.批次?'批次 '+s.批次+' ｜ 状态、成品与检查报告已保存到“运行记录”':''}async function action(url){if(url==='/run')url+='?skip='+document.querySelector('#skip').checked+'&missing='+document.querySelector('#missing').checked;let r=await fetch(url,{method:'POST'});let s=await r.json();if(s.错误)alert(s.错误);await render()}render();</script>"""
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_GET(self):
  if self.path=='/state':data=json.dumps(load(),ensure_ascii=False).encode();kind='application/json'
  elif self.path.startswith('/files/'):
   from urllib.parse import unquote
   p=(ROOT/unquote(self.path[7:])).resolve()
   if not p.is_relative_to(ROOT) or not p.is_file():self.send_error(404);return
   data=p.read_bytes();kind='text/html' if p.suffix=='.html' else 'image/svg+xml'
  else:data=PAGE.encode();kind='text/html'
  self.send_response(200);self.send_header('Content-Type',kind+'; charset=utf-8');self.end_headers();self.wfile.write(data)
 def do_POST(self):
  from urllib.parse import urlparse,parse_qs
  q=urlparse(self.path);a=parse_qs(q.query)
  try:
   if q.path=='/run':s=run(a.get('skip',['false'])[0]=='true',a.get('missing',['false'])[0]=='true')
   elif q.path=='/check':s=external()
   elif q.path=='/approve':s=approve(int(a['n'][0]),'本机演示确认（不代表作者发布批准）')
   else:raise ValueError('未知操作')
  except Exception as e:s={'错误':str(e)}
  self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.end_headers();self.wfile.write(json.dumps(s,ensure_ascii=False).encode())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--运行',action='store_true');p.add_argument('--漏第二次',action='store_true');p.add_argument('--缺规格',action='store_true');p.add_argument('--外部核对',action='store_true');p.add_argument('--端口',type=int,default=8768);a=p.parse_args()
 if a.运行:print(json.dumps(run(a.漏第二次,a.缺规格),ensure_ascii=False))
 elif a.外部核对:print(json.dumps(external(),ensure_ascii=False))
 else:print('打开 http://127.0.0.1:'+str(a.端口),flush=True);HTTPServer(('127.0.0.1',a.端口),Handler).serve_forever()

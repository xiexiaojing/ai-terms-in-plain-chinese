"""把现有漫画画面与文案排成独立SVG和可编辑HTML。无需模型接口。"""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--规格', default='规格.txt')
p.add_argument('--文案', default='文案.json')
p.add_argument('--输出', default='20261004实跑/成品-v1')
a = p.parse_args()
start = time.perf_counter()
required = [ROOT/'原故事.txt', ROOT/a.规格, ROOT/a.文案,
            ROOT/'20261004实跑/画面底图-v1.png']
missing = [f.name for f in required if not f.is_file()]
if missing:
    print(json.dumps({'状态': '停止', '缺少': missing, '本轮生成成品': False}, ensure_ascii=False))
    sys.exit(2)

copy = json.loads(required[2].read_text())
if any(not str(t).strip() for t in copy.get('旁白', [])+copy.get('台词', [])):
    print(json.dumps({'状态':'停止','原因':'旁白或台词缺项','本轮生成成品':False},ensure_ascii=False))
    sys.exit(2)
assert len(copy['旁白']) == 4 and len(copy['台词']) == 4, '文案必须有4条旁白和4条台词'
art = 'data:image/png;base64,' + base64.b64encode(required[3].read_bytes()).decode()
captions = [(364, 60), (1084, 60), (364, 587), (1084, 587)]
speech = [(1138, 150), (836, 201), (985, 201), (425, 674)]

def text_node(text, x, y, size=29, wrap=99):
    chunks = [text[i:i+wrap] for i in range(0, len(text), wrap)]
    first_y = y - (len(chunks)-1)*size*.62
    spans = ''.join(f'<tspan x="{x}" y="{first_y+i*size*1.24:.1f}">{html.escape(c)}</tspan>' for i,c in enumerate(chunks))
    return f'<text letter-spacing="-0.8" text-anchor="middle" font-size="{size}" fill="#202020">{spans}</text>'

nodes = [text_node(t,*pos) for t,pos in zip(copy['旁白'], captions)]
nodes += [text_node(t,*pos,size=26,wrap=7 if i in (0,3) else 99)
          for i,(t,pos) in enumerate(zip(copy['台词'],speech))]
svg = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1448 1086" width="1448" height="1086">'
svg += f'<image width="1448" height="1086" href="{art}"/>'
svg += '<g font-family="PingFang SC, Microsoft YaHei, sans-serif">' + ''.join(nodes) + '</g></svg>'
out = ROOT/a.输出
out.parent.mkdir(parents=True, exist_ok=True)
out.with_suffix('.svg').write_text(svg)
inputs = ''.join(f'<label>第{i+1}格旁白<input data-n="{i}" value="{html.escape(t,quote=True)}"></label>' for i,t in enumerate(copy['旁白']))
page = '''<!DOCTYPE html><meta charset="utf-8"><title>班里没人认识我 四格漫画</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f3f2ef;color:#252525;font-family:"PingFang SC",sans-serif}main{display:grid;grid-template-columns:280px 1fr;gap:24px;padding:26px;max-width:1740px;margin:auto}aside{background:white;padding:22px;border-radius:12px;align-self:start}h1{font-size:22px;margin:0 0 12px}p{font-size:14px;line-height:1.7;color:#666}label{display:block;font-size:14px;margin:18px 0}input{width:100%;font:inherit;padding:10px;margin-top:8px;border:1px solid #ccc;border-radius:5px}button{padding:10px 18px;background:#222;color:white;border:0;border-radius:5px;cursor:pointer}#comic{background:white;line-height:0}svg{width:100%;height:auto;display:block}@media(max-width:800px){main{display:block}aside{margin-bottom:16px}}</style>
<main><aside><h1>班里没人认识我</h1><p>画面与文字分开保存。修改旁白，右侧立即更新。</p>''' + inputs + '''<button id="export">导出当前SVG</button></aside><section id="comic">''' + svg + '''</section></main>
<script>document.querySelectorAll('input').forEach(input=>input.addEventListener('input',()=>{const n=+input.dataset.n;document.querySelectorAll('svg text')[n].querySelector('tspan').textContent=input.value}));document.querySelector('#export').onclick=()=>{const s=new XMLSerializer().serializeToString(document.querySelector('svg'));const url=URL.createObjectURL(new Blob([s],{type:'image/svg+xml'}));const link=document.createElement('a');link.href=url;link.download='班里没人认识我.svg';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};</script>'''
out.with_suffix('.html').write_text(page)
record = {'状态':'已排版，待看图核验','读取文件':[{ '文件':str(f.relative_to(ROOT)), 'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in required],
          '输出':[out.with_suffix('.svg').name, out.with_suffix('.html').name], '本轮模型调用':0,
          '程序耗时秒':round(time.perf_counter()-start,5),'旁白条数':4,'台词条数':4}
out.with_suffix('.运行.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
print(json.dumps(record,ensure_ascii=False))

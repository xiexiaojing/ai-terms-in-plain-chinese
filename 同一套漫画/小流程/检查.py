"""独立读取实际成品；不调用模型，不代替看图。"""
from pathlib import Path
import sys,json,hashlib,xml.etree.ElementTree as ET
p=Path(sys.argv[1]); source=Path(sys.argv[2]); expected=json.loads(source.read_text());root=ET.parse(p).getroot()
actual=[''.join(t.itertext()) for t in root.findall('.//{http://www.w3.org/2000/svg}text')]
want=expected['旁白']+expected['台词']; ok=actual==want
report={'文字检查通过':ok,'实际文字':actual,'预期文字':want,'成品SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'检查进程':__import__('os').getpid(),'看图核验':'待人工确认'}
print(json.dumps(report,ensure_ascii=False));sys.exit(0 if ok else 2)

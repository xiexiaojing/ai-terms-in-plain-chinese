from pathlib import Path
import json, html

folder = Path(__file__).resolve().parent
original = (folder / '原稿.txt').read_text(encoding='utf-8')
lines = original.strip().splitlines()
edited = original.replace('勇于吃亏', '愿意承担')
(folder / '原稿副本.txt').write_text(edited, encoding='utf-8')
result = 12 * 7
(folder / '计算结果.txt').write_text(str(result) + '\n', encoding='utf-8')
cards = ''.join('<li>' + html.escape(line) + '</li>' for line in edited.strip().splitlines()[1:])
page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>合作原则</title><style>body{font:24px sans-serif;max-width:800px;margin:60px auto;padding:24px}li{padding:20px;border-bottom:1px solid #ddd}</style><h1>' + html.escape(lines[0]) + '</h1><ol>' + cards + '</ol></html>'
(folder / '一页展示.html').write_text(page, encoding='utf-8')
record = {'输入': '原稿.txt', '改动': '仅副本：勇于吃亏→愿意承担', '计算': {'表达式': '12 × 7', '结果': result}, '输出': ['原稿副本.txt', '计算结果.txt', '一页展示.html']}
(folder / '执行结果.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(record, ensure_ascii=False))

"""检查已保存的判定记录；不能代替事实核验。仅全部通过才复制输出。"""
import json,sys
from pathlib import Path

def gate(record):
    if record.get('评分标准') != 'rubric-v1': raise ValueError('评分标准版本不匹配')
    rows=record.get('简介',[])
    if len(rows)!=3 or {r.get('id') for r in rows}!={'T1','T2','T3'}: raise ValueError('必须有T1、T2、T3且不能重复')
    for row in rows:
        name=row['id'];text=row.get('文字','')
        if not isinstance(text,str) or not text.strip(): raise ValueError(name+'缺少实际简介')
        if len(''.join(text.split()))>80 or len(text.strip().splitlines())!=1: raise ValueError(name+'长度或段落不符合要求')
        if row.get('人工核验') is not True: raise ValueError(name+'尚未完成人工核验')
        for key in ['R1','R2','R3','R4']:
            if row.get('判定',{}).get(key)!='通过': raise ValueError(name+'/'+key+'未通过或没有判定')
            evidence=row.get('依据',{}).get(key)
            if not isinstance(evidence,str) or not evidence.strip(): raise ValueError(name+'/'+key+'缺少依据')
    return rows

def main():
    if len(sys.argv)!=3: print('用法：python3 发布门禁.py 记录.json 新的发布目录');return 2
    try:
        source=Path(sys.argv[1]);record=json.loads(source.read_text(encoding='utf-8'));rows=gate(record)
        target=Path(sys.argv[2])
        if target.exists(): raise ValueError('发布目录已存在；请用新的目录，避免覆盖旧结果')
        target.mkdir(parents=True)
        (target/'发布简介.txt').write_text('\n'.join(r['id']+'：'+r['文字'] for r in rows)+'\n',encoding='utf-8')
        (target/'放行记录.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print('阻止发布：'+str(exc));return 1
    print('已放行：'+str(target));return 0

if __name__=='__main__':sys.exit(main())

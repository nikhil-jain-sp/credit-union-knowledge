import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]+'/per_question.jsonl')]
ev={json.loads(l)['id']:json.loads(l) for l in open('eval/public_eval.jsonl')}
for r in rows:
    if r['score']>=0.99 and '-a' not in sys.argv: continue
    e=ev[r['id']]
    print(f"### {r['id']} {r['category']} score={r['score']} miss={r['missed']} trap={r['trap_hits']} src={r['source_hit']}")
    print('Q:',e['question'],'| as_of',e.get('as_of'))
    print('EXP:',e['expected_answer'][:200])
    print('ANS:',r['answer']['text'][:500].replace('\n',' '))
    print('SRC:',[s['url'] for s in r['answer']['sources']][:4])

import json
from tqdm.auto import tqdm


def g_json(file, out_file,  n=1000):
    with open(file, 'r') as f:
        res = json.load(f)

    res_final = []
    for _ in tqdm(range(n), total=n):
        res_final.extend(res)

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(res_final, f, indent=4, ensure_ascii=False)

    print(f'repeat {n} times {file} --->> {out_file} ')

if __name__ == '__main__':
    file = 'personal_assistant.jsonl'
    out_file = 'personal_assistant_final.jsonl'
    g_json(file, out_file,  n=20)



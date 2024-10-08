
from utilities import *


intents = {
    "intents": []
}

print('Abuse' in 'Abuse001_x264.mp4')

label = "Test"
data: dict = load_json(f"./json/UCFCrime_{label}.json")

keys = list(data.keys())
keys = sorted(list(set([extract_text_before_number(k) for k in keys])))

print(keys)

temps = {}

for k in keys:
    temps[k] = {
        "tag": k,
        "patterns": [],
        "responses": []
    }


i = 0
for k, v in data.items():
    # if i == 10:
    #     break

    print('-'*50)
    print('k:', k)

    tag = extract_text_before_number(k)
    if tag is None and tag not in keys:
        print(f"Skipping {k}")
        continue

    temps[tag]["patterns"].extend(v.get("sentences"))
    temps[tag]["responses"].append(k)

    # i += 1

for k in keys:
    # print(k)
    intents["intents"].append(temps[k])
    
save_json(intents, f"./json/intents_{label}.json")

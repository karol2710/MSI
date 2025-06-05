import numpy as np
import pandas as pd


# Do wyrzucenia są kolumny: name[0], languages[4], senses[47], attributes[48], actions[49], legendary_actions[50], source[52]
# Ponieważ zawierają dane albo kompletnie nieistotne jak source, albo są nie do przedstawienia numerycznego.

# df = pd.read_csv("aidedd_blocks2.csv", delimiter=",", quotechar='"', encoding="utf-8")

# df.drop(df.columns[[0, 4, 47, 48, 49, 50, 52]], axis=1, inplace=True)

# df.to_csv("data.csv", index=False)


# df = pd.read_csv("data.csv", delimiter=",", encoding="utf-8")

# df["type"] = df["type"].str.split().str[0]

# df.to_csv("data_V2.csv", index=False)

data = np.loadtxt('data_V2.csv', dtype = 'object', delimiter = ',')

newData = np.delete(data, 0, 0)

size_map = {
    "tiny": 1,
    "small": 2,
    "medium": 3,
    "large": 4,
    "huge": 5,
    "gargantuan": 6
}

type_map = {
    "aberration": 1,
    "beast": 2,
    "celestial": 3,
    "construct": 4,
    "dragon": 5,
    "elemental": 6,
    "fey": 7,
    "fiend": 8,
    "giant": 9,
    "humanoid": 10,
    "monstrosity": 11,
    "ooze": 12,
    "plant": 13,
    "undead": 14
}

alignment_map = {
    "lawful good": 1,
    "neutral good": 2,
    "chaotic good": 3,
    "lawful neutral": 4,
    "neutral": 5,
    "chaotic neutral": 6,
    "lawful evil": 7,
    "neutral evil": 8,
    "chaotic evil": 9,
    "any alignment": 10,
    "unaligned": 11
}

newData[:, 0] = [size_map.get(val, val) for val in newData[:, 0]]

newData[:, 1] = [type_map.get(val, val) for val in newData[:, 1]]

newData[:, 2] = [alignment_map.get(val, val) for val in newData[:, 2]]

np.savetxt("clean_data.csv", newData, delimiter=",", fmt="%s")
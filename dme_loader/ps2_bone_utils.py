HIERARCHY = {
    "WORLDROOT": None,
    "COG": "WORLDROOT",
    "PELVIS": "COG",
    "SPINELOWER": "PELVIS",
    "SPINEMIDDLE": "SPINELOWER",
    "SPINEUPPER": "SPINEMIDDLE",
    "NECK": "SPINEUPPER",
    "HEAD": "NECK",
    "HEAD_END": "HEAD",
    "R_HIP": "PELVIS",
    "R_KNEE": "R_HIP",
    "R_ANKLE": "R_KNEE",
    "R_BALL": "R_ANKLE",
    "R_TOE": "R_BALL",
    "R_CLAVICLE": "SPINEUPPER",
    "R_SHOULDER": "R_CLAVICLE",
    "R_SHOULDERROLL": "R_SHOULDER",
    "R_ELBOW": "R_SHOULDERROLL",
    "R_FOREARM": "R_ELBOW",
    "R_WRIST": "R_FOREARM",
    "R_WEAPON": "R_WRIST",
    "R_THUMBA": "R_WRIST",
    "R_THUMBB": "R_THUMBA",
    "R_THUMBC": "R_THUMBB",
    "R_THUMB_END": "R_THUMBC",
    "R_MIDDLEA": "R_WRIST",
    "R_MIDDLEB": "R_MIDDLEA",
    "R_MIDDLEC": "R_MIDDLEB",
    "R_MIDDLE_END": "R_MIDDLEC",
    "R_INDEXA": "R_WRIST",
    "R_INDEXB": "R_INDEXA",
    "R_INDEXC": "R_INDEXB",
    "R_INDEX_END": "R_INDEXC",
    "R_RINGA": "R_WRIST",
    "R_RINGB": "R_RINGA",
    "R_RINGC": "R_RINGB",
    "R_RING_END": "R_RINGC",
    "R_PINKYA": "R_WRIST",
    "R_PINKYB": "R_PINKYA",
    "R_PINKYC": "R_PINKYB",
    "R_PINKY_END": "R_PINKYC",
    "L_HIP": "PELVIS",
    "L_KNEE": "L_HIP",
    "L_ANKLE": "L_KNEE",
    "L_BALL": "L_ANKLE",
    "L_TOE": "L_BALL",
    "L_CLAVICLE": "SPINEUPPER",
    "L_SHOULDER": "L_CLAVICLE",
    "L_SHOULDERROLL": "L_SHOULDER",
    "L_ELBOW": "L_SHOULDERROLL",
    "L_FOREARM": "L_ELBOW",
    "L_WRIST": "L_FOREARM",
    "L_WEAPON": "L_WRIST",
    "L_THUMBA": "L_WRIST",
    "L_THUMBB": "L_THUMBA",
    "L_THUMBC": "L_THUMBB",
    "L_THUMB_END": "L_THUMBC",
    "L_MIDDLEA": "L_WRIST",
    "L_MIDDLEB": "L_MIDDLEA",
    "L_MIDDLEC": "L_MIDDLEB",
    "L_MIDDLE_END": "L_MIDDLEC",
    "L_INDEXA": "L_WRIST",
    "L_INDEXB": "L_INDEXA",
    "L_INDEXC": "L_INDEXB",
    "L_INDEX_END": "L_INDEXC",
    "L_RINGA": "L_WRIST",
    "L_RINGB": "L_RINGA",
    "L_RINGC": "L_RINGB",
    "L_RING_END": "L_RINGC",
    "L_PINKYA": "L_WRIST",
    "L_PINKYB": "L_PINKYA",
    "L_PINKYC": "L_PINKYB",
    "L_PINKY_END": "L_PINKYC",
}

RIGIFY_MAPPINGS = {
    "PELVIS": "DEF-spine",
    "SPINELOWER": "DEF-spine.001",
    "SPINEMIDDLE": "DEF-spine.002",
    "SPINEUPPER": "DEF-spine.003",
    "NECK": "DEF-spine.005",
    "HEAD": "DEF-spine.006",
    "R_HIP": "DEF-thigh.R",
    "R_KNEE": "DEF-shin.R",
    "R_ANKLE": "DEF-foot.R",
    "R_BALL": "DEF-toe.R",
    "R_CLAVICLE": "DEF-shoulder.R",
    "R_SHOULDER": "DEF-upper_arm.R",
    "R_SHOULDERROLL": "DEF-upper_arm.R.001",
    "R_ELBOW": "DEF-forearm.R",
    "R_FOREARM": "DEF-forearm.R.001",
    "R_WRIST": "DEF-hand.R",
    "R_THUMBA": "DEF-thumb.01.R",
    "R_THUMBB": "DEF-thumb.02.R",
    "R_THUMBC": "DEF-thumb.03.R",
    "R_MIDDLEA": "DEF-f_middle.01.R",
    "R_MIDDLEB": "DEF-f_middle.02.R",
    "R_MIDDLEC": "DEF-f_middle.03.R",
    "R_INDEXA": "DEF-f_index.01.R",
    "R_INDEXB": "DEF-f_index.02.R",
    "R_INDEXC": "DEF-f_index.03.R",
    "R_RINGA": "DEF-f_ring.01.R",
    "R_RINGB": "DEF-f_ring.02.R",
    "R_RINGC": "DEF-f_ring.03.R",
    "R_PINKYA": "DEF-f_pinky.01.R",
    "R_PINKYB": "DEF-f_pinky.02.R",
    "R_PINKYC": "DEF-f_pinky.03.R",
    "L_HIP": "DEF-thigh.L",
    "L_KNEE": "DEF-shin.L",
    "L_ANKLE": "DEF-foot.L",
    "L_BALL": "DEF-toe.L",
    "L_CLAVICLE": "DEF-shoulder.L",
    "L_SHOULDER": "DEF-upper_arm.L",
    "L_SHOULDERROLL": "DEF-upper_arm.L.001",
    "L_ELBOW": "DEF-forearm.L",
    "L_FOREARM": "DEF-forearm.L.001",
    "L_WRIST": "DEF-hand.L",
    "L_THUMBA": "DEF-thumb.01.L",
    "L_THUMBB": "DEF-thumb.02.L",
    "L_THUMBC": "DEF-thumb.03.L",
    "L_MIDDLEA": "DEF-f_middle.01.L",
    "L_MIDDLEB": "DEF-f_middle.02.L",
    "L_MIDDLEC": "DEF-f_middle.03.L",
    "L_INDEXA": "DEF-f_index.01.L",
    "L_INDEXB": "DEF-f_index.02.L",
    "L_INDEXC": "DEF-f_index.03.L",
    "L_RINGA": "DEF-f_ring.01.L",
    "L_RINGB": "DEF-f_ring.02.L",
    "L_RINGC": "DEF-f_ring.03.L",
    "L_PINKYA": "DEF-f_pinky.01.L",
    "L_PINKYB": "DEF-f_pinky.02.L",
    "L_PINKYC": "DEF-f_pinky.03.L",
}

BONE_HASHMAP = {
    2726916658: "WORLDROOT",
    567791785: "COG",
    3061072950: "SPINEUPPER",
    4179454272: "NECK",
    898299046: "HEAD",
    233509630: "HEAD_END",
    4217541411: "L_CLAVICLE",
    113224493: "R_CLAVICLE",
    3783973932: "R_KNEE",
    4268096845: "R_ANKLE",
    2049782909: "R_BALL",
    1939367865: "R_TOE",
    2907905148: "R_HIP",
    2072860801: "PELVIS",
    3356280828: "R_FOREARM",
    3360970358: "R_SHOULDERROLL",
    4060706981: "SPINELOWER",
    2242507051: "SPINEMIDDLE",
    1665358410: "R_SHOULDER",
    2970756967: "R_ELBOW",
    1310993196: "R_WRIST",
    2499881490: "R_WEAPON",
    1102577659: "R_THUMBA",
    787765876: "R_THUMBB",
    3860416703: "R_THUMBC",
    2176474540: "R_THUMB_END",
    812161446: "R_MIDDLEA",
    1034073114: "R_MIDDLEB",
    148130434: "R_MIDDLEC",
    3546037191: "R_MIDDLE_END",
    3770474639: "R_INDEXA",
    4277804297: "R_INDEXB",
    2054689799: "R_INDEXC",
    2492699346: "R_INDEX_END",
    931042330: "R_RINGA",
    1237039252: "R_RINGB",
    2624511473: "R_RINGC",
    2819542915: "R_RING_END",
    1719789456: "R_PINKYA",
    869597751: "R_PINKYB",
    3693662948: "R_PINKYC",
    1729498404: "R_PINKY_END",
    1900803227: "L_HIP",
    524728837: "L_KNEE",
    2525539168: "L_ANKLE",
    1142326112: "L_BALL",
    2357602519: "L_TOE",
    3746558302: "L_SHOULDER",
    3291028981: "L_SHOULDERROLL",
    2266553668: "L_ELBOW",
    4157307685: "L_FOREARM",
    3216727789: "L_WRIST",
    569531117: "L_WEAPON",
    978245741: "L_THUMBA",
    1282964672: "L_THUMBB",
    1453035782: "L_THUMBC",
    1965938483: "L_THUMB_END",
    724094749: "L_MIDDLEA",
    198244558: "L_INDEXA",
    943876432: "L_MIDDLEB",
    3337782962: "L_MIDDLEC",
    810894083: "L_MIDDLE_END",
    4212840290: "L_INDEXB",
    3930207657: "L_INDEXC",
    2209626904: "L_INDEX_END",
    3662356388: "L_RINGA",
    3893213993: "L_RINGB",
    2769958247: "L_RINGC",
    1143123657: "L_RING_END",
    3483479177: "L_PINKYA",
    127966342: "L_PINKYB",
    887355148: "L_PINKYC",
    3920976780: "L_PINKY_END",
}
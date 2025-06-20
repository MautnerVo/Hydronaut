cz_to_en = {
    'postavení chodidel': 'foot_position',
    'směr špiček': 'toe_direction',
    'chodidla': 'feet',
    'osa hlezenních kloubů': 'ankle_joint_axis',
    'hloubka dřepu': 'squat_depth',
    'osa kolenních kloubů': 'knee_joint_axis',
    'napřímení páteře': 'spine_alignment',
    'osa hlavy': 'head_axis',
    'plynulost pohybu': 'movement_fluidity',
    'technika': 'technique',
    'tempo': 'tempo',
    'stabilita': 'stability',
    'postavení DKK': 'lower_limb_position',
    'těžiště': 'center_of_gravity',
    'plná extenze KYK': 'full_hip_extension',
    'postavení dlaní': 'hand_position',
    'směr prstů': 'finger_direction',
    'postavení loktů': 'elbow_position',
    'hloubka kliku': 'pushup_depth',
    'plný rozsah pohybu': 'full_range_of_motion',
    'úhel v kyčli': 'hip_angle',
    'úhel opory': 'support_angle',
    'extenze loktů': 'elbow_extension',
    'rovnoměrný pohyb konč.': 'limb_movement_uniformity',
    'postavení rukou': 'arm_position',
    'postavení nohou': 'leg_position',
    'symetrie pohybu': 'movement_symmetry',
    'linie těla': 'body_line',
    'rozsah pohybu': 'range_of_motion',
    'výška výskoku': 'jump_height',
    'přechod do dřepu I': 'transition_to_squat_I',
    'přechod do vzporu I': 'transition_to_plank_I',
    'pozice ve vzporu': 'plank_position',
    'položení těla na zem': 'body_down_to_floor',
    'přechod do vzporu II': 'transition_to_plank_II',
    'přechod do dřepu II': 'transition_to_squat_II',
    'výdrž a tempo': 'hold_and_tempo',
    'stabilita trupu I': 'core_stability_I',
    'stabilita trupu II': 'core_stability_II',
    'poloha nohou': 'leg_placement',
    'doskok': 'landing',
    'stabilita trupu': 'core_stability',
    'poloha vzporu': 'plank_alignment',
    'rozložení váhy': 'weight_distribution',
    'úhel kyčlí': 'hip_joint_angle'
}
bool_dict = {"ano":1,
             "ne":0}
exercise = {
    "SQUAT": "Standard squat",
    "SQUAT - hold": "Squat hold",
    "WIDE SQUAT": "Wide squat",
    "WIDE SQUAT - hold": "Wide squat hold",
    "LUNGE": "Lunge",
    "LUNGE - hold": "Lunge hold",
    "BRIDGING": "Bridging",
    "BRIDGING - hold": "Bridging hold",
    "PUSH-UP": "Push-ups",
    "FOREARM PLANK": "Forearm plank hold",
    "TRICEPS PUSH-UP": "Triceps push-ups",
    "EXT. ARM PLANK": "Extended arm plank hold",
    "SUPERMAN": "Superman",
    "SUPERMAN hold": "Superman hold",
    "SIDE PLANK ROT.": "Side plank rotation",
    "SIDE PLANK hold": "Side plank hold",
    "BURPEES": "Burpees",
    "PLANK WALKOUT": "Plank walkout",
    "JUMP SQUAT": "Jump squats",
    "MOUNTAIN CLIMBERS": "Mountain climbers"
}


import pandas as pd
import os
folder = r"Y:\Datasets\Fyzio\2025-03-07\2"
file = r"2_hodnoceni.xlsx"
path = os.path.join(folder, file)

df = pd.read_excel(path,header=None)

ended = True


tables = []
start = 0
coll = 3
for index,B in enumerate(df.iloc[5:,1],start=5):
    if not pd.isna(B) and pd.isna(df.iloc[index,2]):
        start = index
        if not pd.isna(df.iloc[index,3]):
            coll = 5
        else:
            coll = 4
        ended = False
    if not ended and pd.isna(B):
        ended = True
        end = index+2
        tables.append((start, end,coll))


data = []
for index,table in enumerate(tables):
    krit = False
    sub_df = df.iloc[table[0]:table[1],1:table[2]]
    sub_data = [[],[]]
    for index,col in enumerate(sub_df.iloc[:,0]):
        if krit and not pd.isna(col):
            sub_data[0].append(cz_to_en[col.rstrip()])
            sub_data[1].append(bool_dict[sub_df.iloc[index,-1]])
        if col == "kritérium":
            krit = True
    sub_data[0].append("overall quality")
    sub_data[1].append(sub_df.iloc[-1,-1])
    data.append(sub_data)

for index, ex_data in enumerate(data):
    df_out = pd.DataFrame.from_records(ex_data)
    os.makedirs("exercises", exist_ok=True)
    name = str(df.iloc[tables[index][0], 1]).strip()
    df_out.to_csv(f"exercises/{exercise[name]}.csv", index=False, header=False)



# print(df.iloc[tables[0][1],3])
# unique = df.iloc[:,1].unique().tolist()
# rm_unique = []
# for table in tables:
#     rm_unique.append(df.iloc[table[0],1])
#
# df_filtered = [x for x in unique if x not in rm_unique]
# print(len(df_filtered) - 2)
# print(df_filtered)
import os
import numpy as np
import pandas as pd

def to_num(x):
    """
    Převede vstupní hodnotu na float, nahradí českou čárku tečkou.
    Pokud převod selže, vrátí None.
    """
    if isinstance(x, str):
        x = x.strip().replace(',', '.')
    try:
        return float(x)
    except Exception:
        return None

def process_file(filepath):
    """
    Načte jeden soubor EMG_anotace*.xlsx, vytáhne pro každou
    aktivitu poslední non-None časy 'Beginning' a 'Ending'
    z prvního datového řádku a vytiskne je.
    """
    # Načtení s hlavičkami na řádcích 3 a 4 (0-based: 2,3)
    df = pd.read_excel(filepath, header=[2, 3])
    raw = df.iloc[0]

    # Převedeme všechny hodnoty na float / None
    nums = raw.apply(to_num)

    annot = {}
    # Projdeme všechny aktivity (úroveň 0 hlavičky)
    for name in df.columns.get_level_values(0).unique():
        if not isinstance(name, str) or pd.isna(name):
            continue

        # všechny sloupce pro tuto aktivitu
        cols = [col for col in df.columns if col[0] == name]

        # vybereme ty, jejichž úroveň 1 začíná na 'Beginning' / 'Ending'
        begins = [c for c in cols if isinstance(c[1], str) and c[1].startswith('Beginning')]
        ends   = [c for c in cols if isinstance(c[1], str) and c[1].startswith('Ending')]

        # sběr hodnot
        b_vals = []
        e_vals = []
        for c in begins:
            val = nums[c]
            if val is not None:
                b_vals.append(val)
        for c in ends:
            val = nums[c]
            if val is not None:
                e_vals.append(val)

        # pokud máme obě hodnoty, uložíme poslední (nejvíc vpravo)
        if b_vals and e_vals:
            annot[name.strip()] = {
                'begin': b_vals[-1],
                'end':   e_vals[-1]
            }

    print(f"\n=== {os.path.basename(filepath)} ===")
    if not annot:
        print("Žádné validní anotace nenalezeny.")
    else:
        filtered_annot = {k: v for k, v in annot.items() if not (np.isnan(v['begin']) or np.isnan(v['end']))}
        return filtered_annot
        # for act, times in filtered_annot.items():
        #     if not pd.isna(times['begin']) and not pd.isna(times['end']):
        #         print(f"annot['{act}']['begin'] = {times['begin']}")
        #         print(f"annot['{act}']['end']   = {times['end']}")

if __name__ == '__main__':
    root_dir = r'E:\Datasets\Fyzio'

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.startswith('EMG_anotace') and fname.lower().endswith('.xlsx'):
                fullpath = os.path.join(dirpath, fname)
                try:
                    process_file(fullpath)
                except Exception as e:
                    print(f"Chyba při zpracování {fname}: {e}")

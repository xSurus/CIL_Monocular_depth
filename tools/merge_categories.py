import argparse
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--aerial',      required=True)
    p.add_argument('--water',       required=True)
    p.add_argument('--vegetation',  required=True)
    p.add_argument('--out',         required=True)
    args = p.parse_args()

    aerial = pd.read_csv(args.aerial)[['file', 'label']].rename(columns={'label': 'aerial_label'})
    water  = pd.read_csv(args.water)[['file', 'label']].rename(columns={'label': 'water_label'})
    veg    = pd.read_csv(args.vegetation)[['file', 'label']].rename(columns={'label': 'vegetation_label'})

    df = aerial.merge(water, on='file').merge(veg, on='file')

    def assign(row):
        if row['aerial_label'] == 'aerial':           return 'aerial'
        if row['water_label'] == 'water':             return 'water'
        if row['vegetation_label'] == 'vegetation':   return 'vegetation'
        return 'other'

    df['category'] = df.apply(assign, axis=1)
    df[['file', 'category']].to_csv(args.out, index=False)

    counts = df['category'].value_counts()
    total  = len(df)
    print('Category distribution:')
    for cat, n in counts.items():
        print(f'  {cat:<12} {n:>6}  ({100*n/total:.1f}%)')
    print(f'  {"TOTAL":<12} {total:>6}')
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()

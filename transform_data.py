import pandas as pd

# CSV-bestanden inlezen
df1 = pd.read_csv('./output/3-10-2024-ochtend-licence.csv')  # Met header
df2 = pd.read_csv('./output/3-10-2024-middag-licence.csv')

# Kolom met kentekens extraheren (kolom met naam 'Kenteken')
kolom_kenteken1 = df1['Kenteken']  # 'Kenteken' kolom uit bestand1
kolom_kenteken2 = df2['Kenteken']  # 'Kenteken' kolom uit bestand2

# Vind de overeenkomende kentekens
overeenkomstige_kentekens = kolom_kenteken1[kolom_kenteken1.isin(kolom_kenteken2)]

# Print de overeenkomende kentekens samen met Merk en Tijden
herkende_kentekens = set()

for kenteken in overeenkomstige_kentekens:
    if kenteken not in herkende_kentekens:
        # Haal de rijen op voor het overeenkomstige kenteken in beide dataframes
        rij_df1 = df1[df1['Kenteken'] == kenteken].iloc[0]
        rij_df2 = df2[df2['Kenteken'] == kenteken].iloc[0]
        
        # Print het kenteken, het merk en de tijden uit beide bestanden
        print(f"Kenteken: {kenteken}")
        print(f"Merk: {rij_df1['Auto merk']} | Tijd1: {rij_df1['Tijd']} | Tijd2: {rij_df2['Tijd']}")
        print("-" * 50)
        
        # Voeg het kenteken toe aan de set van herkende kentekens
        herkende_kentekens.add(kenteken)
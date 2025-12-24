import pandas as pd

# Data: Username (from filename) mapped to Leaning
# Classifications based on public bio, content alignment, and audience overlap.
# "Neutral" includes objective journalists, niche topics (Space, History), or foreign analysts not aligning with US binary.

data = {
    'post_id': [
        'AdamParkhomenko', 'ArielleScarcell', 'AstroTerry', 'bariweiss', 'blackintheempir',
        'bresreports', 'brithume', 'burgessev', 'ByYourLogic', 'CarlHigbie',
        'CensoredMen', 'ChrisLoesch', 'CollinRugg', 'danpfeiffer', 'danprimack',
        'dlacalle_IA', 'DrLoupis', 'elonmusk', 'eveforamerica', 'FiorellaIsabelM',
        'FrankDangelo23', 'garethicke', 'GregRubini', 'HilzFuld', 'IamBrookJackson',
        'jayrosen_nyu', 'jimsciutto', 'JoeConchaTV', 'JonahDispatch', 'JonathanTurley',
        'JoshDenny', 'kacdnp91', 'KatiePhang', 'kristina_wong', 'kyledcheney',
        'laurashin', 'LEBassett', 'LeftAtLondon', 'Leslieoo7', 'LibertyCappy',
        'MarchandSurgery', 'MarkHertling', 'MaryLTrump', 'MattBruenig', 'MikeASperrazza',
        'MikeSington', 'molly0xFFF', 'MsAvaArmstrong', 'NAChristakis', 'omgno2trump',
        'Prolotario1', 'Rach_IC', 'realDonaldTrump', 'RightWingCope', 'RogerJStoneJr',
        'saifedean', 'SamParkerSenate', 'SarahTheHaider', 'secupp', 'simon_schama',
        'StellaParton', 'Tatarigami_UA', 'thatdayin1992', 'TheBigMigShow', 'thecoastguy',
        'thejackhopkins', 'timinhonolulu', 'WBrettWilson', 'WhiteHouse', 'yarahawari',
        'ylecun'
    ],
    'leaning': [
        'Left', 'Right', 'Neutral', 'Right', 'Left', 
        'Neutral', 'Right', 'Neutral', 'Left', 'Right', 
        'Right', 'Right', 'Right', 'Left', 'Neutral', 
        'Right', 'Right', 'Right', 'Right', 'Right', # FiorellaIsabelM functions in MAGA ecosystem
        'Right', 'Right', 'Right', 'Right', 'Right', 
        'Left', 'Left', 'Right', 'Right', 'Right', 
        'Right', 'Right', 'Left', 'Right', 'Neutral', 
        'Neutral', 'Left', 'Left', 'Right', 'Right', 
        'Left', 'Left', 'Left', 'Left', 'Right', 
        'Left', 'Left', 'Right', 'Neutral', 'Left', 
        'Right', 'Left', 'Right', 'Left', 'Right', 
        'Right', 'Right', 'Right', 'Right', 'Left', 
        'Left', 'Neutral', 'Neutral', 'Right', 'Right', 
        'Left', 'Left', 'Right', 'Left', 'Left', 
        'Left'
    ]
}

# Create DataFrame
df_leaning = pd.DataFrame(data)

# Save to CSV
output_path = 'influencer_leaning.csv'
df_leaning.to_csv(output_path, index=False)

print(f"File '{output_path}' created successfully with {len(df_leaning)} entries.")
print("\nBreakdown:")
print(df_leaning['leaning'].value_counts())
import pandas as pd
import numpy as np
import ast
import re


TOP_100_ACTORS = [
    'nm0482320', 'nm0007123', 'nm0103977', 'nm0695177', 'nm0000616',
    'nm0621937', 'nm0004429', 'nm0451600', 'nm0000821', 'nm0474774',
    'nm0352032', 'nm0149822', 'nm0415549', 'nm0080238', 'nm0893449',
    'nm0419688', 'nm0707425', 'nm0700875', 'nm0000367', 'nm0035067',
    'nm0712546', 'nm0004569', 'nm0007106', 'nm0000514', 'nm0595934',
    'nm0006369', 'nm0004109', 'nm0019382', 'nm1388202', 'nm0159159',
    'nm0006763', 'nm0000465', 'nm0222426', 'nm0611481', 'nm0004434',
    'nm0000115', 'nm0000134', 'nm0000353', 'nm0490489', 'nm0420090',
    'nm0000246', 'nm0438463', 'nm0000168', 'nm0787462', 'nm0226770',
    'nm0869451', 'nm0158112', 'nm0001595', 'nm0945189', 'nm0000329',
    'nm0000661', 'nm0409204', 'nm0043199', 'nm0000078', 'nm0000323',
    'nm0938893', 'nm0006795', 'nm0820208', 'nm0465503', 'nm0000489',
    'nm0332871', 'nm0700869', 'nm1335387', 'nm0438501', 'nm2147526',
    'nm0000104', 'nm0451396', 'nm0004435', 'nm6489058', 'nm0154164',
    'nm0001000', 'nm0695199', 'nm0001744', 'nm0049395', 'nm0579756',
    'nm1428724', 'nm0015459', 'nm0001136', 'nm0329730', 'nm0000448',
    'nm0865575', 'nm0018203', 'nm0000172', 'nm0634159', 'nm0792911',
    'nm0003501', 'nm0814773', 'nm0874676', 'nm0000052', 'nm0681566',
    'nm0045075', 'nm0088396', 'nm0000553', 'nm0219971', 'nm0000809',
    'nm0474876', 'nm0304262', 'nm0004417', 'nm0001016', 'nm0839017'
]


def data_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    מקבלת DataFrame גולמי של סרטים ומחזירה DataFrame מעובד ומוכן למודל.
    """
    df = df.copy()

    # 1. הסרת עמודות Data Leakage ולא רלוונטיות
    cols_to_drop = ['plot', 'numVotes', 'BoxOffice']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 2. averageRating
    if 'averageRating' in df.columns:
        df = df[pd.to_numeric(df['averageRating'], errors='coerce').notna() | df['averageRating'].isna()]
        df['averageRating'] = pd.to_numeric(df['averageRating'], errors='coerce')

    # 3. startYear → decade
    if 'startYear' in df.columns:
        df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce')
        df['startYear'] = df['startYear'].apply(
            lambda x: np.nan if pd.isna(x) or x == 0 else int(x)
        )
        df['decade'] = df['startYear'].apply(
            lambda x: int((x // 10) * 10) if pd.notna(x) else -1
        )
        df = df.drop(columns=['startYear'])

    # 4. genres
    if 'genres' in df.columns:
        def clean_genres(g):
            if pd.isna(g) or str(g).strip() in ('[]', r'\N', ''):
                return 'Unknown'
            try:
                parsed = ast.literal_eval(g)
                if isinstance(parsed, list):
                    return ','.join([x.strip() for x in parsed if x.strip()])
            except:
                pass
            return str(g).strip()

        df['genres'] = df['genres'].apply(clean_genres)
        df['genres'] = df['genres'].replace(r'\N', 'Unknown')

        top_16 = ['Drama', 'Comedy', 'Romance', 'Action', 'Documentary',
                  'Crime', 'Thriller', 'Horror', 'Adventure', 'Mystery',
                  'Family', 'Fantasy', 'Biography', 'History', 'Sci-Fi', 'Music']

        for genre in top_16:
            df[genre] = df['genres'].str.contains(genre, na=False).astype(int)

        df['genre_Unknown'] = (df['genres'] == 'Unknown').astype(int)

        other_genres_list = ['Musical', 'War', 'Animation', 'Sport', 'Western',
                             'Adult', 'Film-Noir', 'News', 'Reality-TV', 'Talk-Show', 'Game-Show']
        df['genre_Other'] = df['genres'].apply(
            lambda x: 1 if any(g in str(x).split(',') for g in other_genres_list) else 0
        )

        df = df.drop(columns=['genres'])

    # 5. lead_actors_ids — Top 100 + Other + Count
    if 'lead_actors_ids' in df.columns:
        def count_actors(x):
            if pd.isna(x) or str(x).strip() in ('Unknown', '[]', ''):
                return -1
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return len(parsed)
            except:
                pass
            return -1

        df['num_lead_actors'] = df['lead_actors_ids'].apply(count_actors)

        for actor in TOP_100_ACTORS:
            df[f'actor_{actor}'] = df['lead_actors_ids'].apply(
                lambda x: 1 if isinstance(x, str) and actor in x else 0
            )

        df['other_actor'] = df['lead_actors_ids'].apply(
            lambda x: 1 if isinstance(x, str)
            and str(x).strip() not in ('Unknown', '[]', '')
            and not any(actor in x for actor in TOP_100_ACTORS) else 0
        )

        df = df.drop(columns=['lead_actors_ids'])

    # 6. runtimeMinutes — בריבוע + missing
    if 'runtimeMinutes' in df.columns:
        df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce')
        df['missing_runtime'] = df['runtimeMinutes'].isna().astype(int)
        df['runtimeMinutes'] = df['runtimeMinutes'].fillna(0)
        df['runtimeMinutes_squared'] = df['runtimeMinutes'] ** 2

    # 7. Language
    if 'Language' in df.columns:
        df['Language'] = df['Language'].fillna('Unknown')
        df['Language'] = df['Language'].replace('Not Found', 'Unknown')

        def clean_language(x):
            if x == 'Unknown':
                return 'Unknown'
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return ','.join([lang.strip() for lang in parsed])
            except:
                pass
            return ','.join([lang.strip() for lang in str(x).split(',')])

        df['Language'] = df['Language'].apply(clean_language)

        def clean_language_special(x):
            if x == 'Unknown':
                return 'Unknown'
            if re.search(r'[^a-zA-Z,\s]', x):
                return 'Unknown'
            return x

        df['Language'] = df['Language'].apply(clean_language_special)

        top_languages = ['English', 'French', 'Hindi', 'Spanish']
        for lang in top_languages:
            df[lang] = df['Language'].apply(
                lambda x: 1 if isinstance(x, str) and lang in x.split(',') else 0
            )
        df['Language_Unknown'] = (df['Language'] == 'Unknown').astype(int)
        df['Language_Other'] = df['Language'].apply(
            lambda x: 1 if x != 'Unknown' and not any(
                lang in str(x).split(',') for lang in top_languages) else 0
        )
        df = df.drop(columns=['Language'])

    # 8. Country
    if 'Country' in df.columns:
        df['Country'] = df['Country'].fillna('Unknown')
        df['Country'] = df['Country'].replace('Not Found', 'Unknown')

        def clean_country(x):
            if x == 'Unknown':
                return 'Unknown'
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return ','.join([c.strip() for c in parsed])
            except:
                pass
            return ','.join([c.strip() for c in str(x).split(',')])

        df['Country'] = df['Country'].apply(clean_country)

        def clean_country_special(x):
            if x == 'Unknown':
                return 'Unknown'
            if re.search(r'[^a-zA-Z,\s]', x):
                return 'Unknown'
            return x

        df['Country'] = df['Country'].apply(clean_country_special)

        top_countries = ['United States', 'India', 'United Kingdom',
                         'France', 'Italy', 'Japan', 'Canada']
        for country in top_countries:
            df[country] = df['Country'].apply(
                lambda x: 1 if isinstance(x, str) and country in x.split(',') else 0
            )
        df['Country_Unknown'] = (df['Country'] == 'Unknown').astype(int)
        df['Country_Other'] = df['Country'].apply(
            lambda x: 1 if x != 'Unknown' and not any(
                country in str(x).split(',') for country in top_countries) else 0
        )
        df = df.drop(columns=['Country'])

    # 9. budget
    if 'budget' in df.columns:
        df['budget'] = df['budget'].fillna('Unknown')
        df['budget'] = df['budget'].replace('0', 'Unknown').replace(0, 'Unknown')
        df['has_budget'] = df['budget'].apply(lambda x: 0 if x == 'Unknown' else 1)
        df = df.drop(columns=['budget'])

    return df
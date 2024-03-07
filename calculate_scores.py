import pandas as pd

df = pd.read_csv('ich_data.csv')

def calculate_score(conditions, scores):
    return sum(score for condition, score in zip(conditions, scores) if condition)

def prognostic_score_oICH(age, hemsize, gcsadm, infratentorial, ivh):
    conditions = [age > 80, hemsize == 4, 5 <= gcsadm <= 12, gcsadm <= 4, infratentorial == 1, ivh == 1]
    scores = [1, 1, 1, 2, 1, 1]
    return calculate_score(conditions, scores)

def prognostic_score_mICH(age, hemsize, nihssadm, infratentorial, ivh):
    conditions = [age > 80, hemsize == 4, 11 <= nihssadm <= 20, nihssadm > 20, infratentorial == 1, ivh == 1]
    scores = [1, 1, 1, 2, 1, 1]
    return calculate_score(conditions, scores)

def prognostic_score_ICH_GS(age, volume1, gcsadm, infratentorial, ivh):
    age_score = 1 if 45 <= age <= 64 else 2 if age >= 65 else 0
    volume_score = 1 if (infratentorial == 1 and 10 <= volume1 <= 20) or (infratentorial == 0 and 40 <= volume1 <= 70) else 2 if volume1 > 20 or (infratentorial == 0 and volume1 > 70) else 0
    gcs_score = 1 if 9 <= gcsadm <= 12 else 2 if gcsadm <= 8 else 0
    extra_score = 5 if age >= 65 and volume1 > 20 and gcsadm <= 8 and infratentorial == 1 and ivh == 1 else 0
    return age_score + volume_score + gcs_score + (1 if infratentorial == 1 else 0) + (1 if ivh == 1 else 0) + extra_score

def prognostic_score_LSICH(hemsize, gcsadm, ivh, dmadm):
    conditions = [hemsize == 4, 5 <= gcsadm <= 12, gcsadm <= 4, ivh == 1, dmadm == 1]
    scores = [1, 1, 2, 1, 1]
    return calculate_score(conditions, scores)

def prognostic_score_ICH_FOS(age, volume1, gcsadm, nihssadm, infratentorial, ivh, rbs):
    age_score = 1 if 60 <= age <= 69 else 2 if 70 <= age <= 79 else 4 if age > 80 else 0
    volume_score = 2 if (infratentorial == 1 and volume1 > 10) or (infratentorial == 0 and volume1 > 40) else 0
    gcs_score = 1 if 9 <= gcsadm <= 12 else 2 if gcsadm <= 8 else 0
    nihss_score = 2 if 6 <= nihssadm <= 10 else 3 if 11 <= nihssadm <= 15 else 4 if 16 <= nihssadm <= 20 else 5 if nihssadm > 20 else 0
    return age_score + volume_score + gcs_score + nihss_score + (1 if infratentorial == 1 else 0) + (1 if ivh == 1 else 0) + (1 if rbs > 11 else 0)


def prognostic_score_max_ICH(age, nihssadm, ivh, priorac, site, volume1):
    age_scores = [(70 <= age <= 74, 1), (75 <= age <= 79, 2), (age >= 80, 3)]
    nihssadm_scores = [(7 <= nihssadm <= 13, 1), (14 <= nihssadm <= 20, 2), (nihssadm >= 21, 3)]
    
    score = sum(score for condition, score in age_scores if condition)
    score += sum(score for condition, score in nihssadm_scores if condition)
    score += ivh + priorac  # Assuming IVH and PRIORAC are binary (0 or 1)

    lobar_sites = [3, 6]
    nonlobar_sites = [1, 2, 4, 5]
    
    if site in lobar_sites and volume1 >= 30:
        score += 1
    elif site in nonlobar_sites and volume1 >= 10:
        score += 1

    return score

# Modifying the functions to accept a row of the DataFrame
def prognostic_score_oICH_row(row):
    return prognostic_score_oICH(age=row['AGE'], hemsize=row['HEMSIZE'], gcsadm=row['GCSADM'], infratentorial=row['INFRATENTORIAL'], ivh=row['IVH'])

def prognostic_score_mICH_row(row):
    return prognostic_score_mICH(age=row['AGE'], hemsize=row['HEMSIZE'], nihssadm=row['NIHSSADM'], infratentorial=row['INFRATENTORIAL'], ivh=row['IVH'])

def prognostic_score_ICH_GS_row(row):
    return prognostic_score_ICH_GS(age=row['AGE'], volume1=row['VOLUME1'], gcsadm=row['GCSADM'], infratentorial=row['INFRATENTORIAL'], ivh=row['IVH'])

def prognostic_score_LSICH_row(row):
    return prognostic_score_LSICH(hemsize=row['HEMSIZE'], gcsadm=row['GCSADM'], ivh=row['IVH'], dmadm=row['DMADM'])

def prognostic_score_ICH_FOS_row(row):
    return prognostic_score_ICH_FOS(age=row['AGE'], volume1=row['VOLUME1'], gcsadm=row['GCSADM'], nihssadm=row['NIHSSADM'], infratentorial=row['INFRATENTORIAL'], ivh=row['IVH'], rbs=row['RBS'])

def prognostic_score_max_ICH_row(row):
    return prognostic_score_max_ICH(age=row['AGE'], nihssadm=row['NIHSSADM'], ivh=row['IVH'], priorac=row['PRIORAC'], site=row['SITE'], volume1=row['VOLUME1'])

df['VOLUME1'] = pd.to_numeric(df['VOLUME1'], errors='coerce')
df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
df['GCSADM'] = pd.to_numeric(df['GCSADM'], errors='coerce')
df['INFRATENTORIAL'] = pd.to_numeric(df['INFRATENTORIAL'], errors='coerce')
df['IVH'] = pd.to_numeric(df['IVH'], errors='coerce')

# Applying the functions to each row of the DataFrame and creating new columns
df['oICH_score'] = df.apply(prognostic_score_oICH_row, axis=1)
df['mICH_score'] = df.apply(prognostic_score_mICH_row, axis=1)
df['ICH_GS_score'] = df.apply(prognostic_score_ICH_GS_row, axis=1)
df['LSICH_score'] = df.apply(prognostic_score_LSICH_row, axis=1)
df['ICH_FOS_score'] = df.apply(prognostic_score_ICH_FOS_row, axis=1)
df['Max_ICH_score'] = df.apply(prognostic_score_max_ICH_row, axis=1)

df.to_csv('ich_data_w_scores.csv')


# Mapping columns from Dataset 1 to Dataset 2
column_mapping = {
    'AGE': 'AGE (F33Q02)',
    'GENDER': 'GENDER F01Q01 (male 1; female 0)',
    'ETHNICITY': 'F01Q03M1; Race: American Indian or Alaska Native, F01Q03M2; asian, F01Q03M3; black/african american, F01Q03M5; white, F01Q03M6; other, F01Q03M7; unknown',
    'MODE': 'NA',
    'NIHSSADM': 'NIHSS at admission is F10 Q22 in Form 10. Use score prior to randomization denoted as " - "',
    'RBS': 'Glucose (mg/dl) (F15Q11)  Use score prior to randomization denoted as " - "',
    'SBP': 'SBP (prior to randomization) (F05Q09a)',
    'DBP': 'DBP (prior to randomization) 9F05Q09b)',
    'HR': 'NA',
    'BMI': 'NA',
    'PRIORAC': 'NA',
    'PRIORAP': 'NA',
    'PRIORANTIHTN': 'PRIOR ANTIHTN  (F04Q01)',
    'PRIORANTIDM': 'PRIOR ANTIDM (F04Q03)',
    'DMADM': 'DM ADM F03Q11 and 12 (type 1 and 2)',
    'HTNADM': 'HTN ADM (F03Q07)',
    'HTNTYPE': 'NA',
    'DYSLIPADM': 'HLD (F03Q09) (yes=1, no=2, unknwon =98)',
    'STROKE': 'STROKE Hx (F03Q01) (defined as prior stroke/TIA Hx)',
    'AF': 'AF (F03Q04)',
    'CAD': 'CAD (F3Q06)',
    'SMOKING': 'SMOKING (F03Q13)  (defined as current=1,former=2,never=3,unknown=4)',
    'DMTYPE': 'NA',
    'HbA1c': 'NA',
    'CHOL': 'NA',
    'TRIG': 'NA',
    'HDL': 'NA',
    'LDL': 'NA',
    'PLT': 'PLT  (x 10^3 / mm^3) (F15Q07)  Use score prior to randomization denoted as " - "',
    'INR': 'INR (F15Q09)  Use score prior to randomization denoted as " - "',
    'APTT': 'APTT (sec) (F15Q08)  Use score prior to randomization denoted as " - "',
    'EVD': 'EVD F07Q05 (called intraventricular catheter received)',
    'HCM': 'HCM F07Q08 (defined as immediate hematoma evac)',
    'COMBINED': 'NA',
    'ANYPROCEDURE': 'NA',
    'SWCARE': 'NA',
    'SWDIRECT': 'NA',
    'ICUCARE': 'ICUCARE, F1Q01 represent Days in ICU --> Made into  1=Yes and 0= no',
    'INTUBATED': 'INTUBATED (used  F08Q04 ; Not F07Q02)',
    'ARETRIALLINE': 'NA',
    'SITE': 'Site of Bleed (F2Q07)  1=BG, 2=thalamus, 3=lobar) (pontine / cerebellar excluded) (Use variable 45 in Qatar: 1=BG, 2=thalamus, use 3 & 6 to count as lobar, exclude pts with 4,5,7)',
    'GCSADM': 'GCSADM F33Q04',
    'VOLUME1': 'VOLUME 1 ( on admission) in mm3 -CTCR Q 12',
    'HEMSTATUS': 'HEM STATUS ON REPEAT CT (0=no change, 1=increased size, 2= decreased size) --> calculate from volume 1 and 2 and change into cm3',
    'GCSSCORE': 'NA',
    'IVH': 'IVH (F33QO3) (1 = present , 0==absent)',
    'VOLUME30': 'NA',
    'INFRATENTORIAL': 'NA',
    'AGE80': 'AGE > 80   (ask Statistician for help)',
    'ICHSCORE': 'NA',
    'MORT90': 'MORT 90 - (F20Q04 MIN)',
    'MRS90': 'MRS90 (0 - mRS 0-3 , 1 - mRS 4-6)  (Form 21 primary outcome )',
    'HEMSIZE': 'NA',
    'BLEEDSITE': 'NA',
    'oICH_score': 'oICH_score  - check score components and see if we can get it from ATACh data',
    'mICH_score': 'mICH_score - check score components and see if we can get it from ATACh data',
    'ICH_GS_score': 'ICH_GS_score  - check score components and see if we can get it from ATACh data',
    'LSICH_score': 'LSICH_score  - check score components and see if we can get it from ATACh data',
    'ICH_FOS_score': 'ICH_FOS_score  - check score components and see if we can get it from ATACh data',
    'Max_ICH_score': 'Max_ICH_score  - check score components and see if we can get it from ATACh data'
}

# Identify columns in Dataset 2 that do not map to Dataset 1
dataset2_columns = set([
    'SUBJECT_ID', 'AGE (F33Q02)', 'GENDER F01Q01 (male 1; female 0)', 'ETHNICITY - at the end of the sheet', 'NIHSS at admission is F10 Q22 in Form 10. Use score prior to randomization denoted as " - "', 'Glucose (mg/dl) (F15Q11)  Use score prior to randomization denoted as " - "', 'SBP (prior to randomization) (F05Q09a)', 'DBP (prior to randomization) 9F05Q09b)', 'PRIOR ANTIHTN  (F04Q01)', 'PRIOR ANTIDM (F04Q03)', 'DM ADM F03Q11 and 12 (type 1 and 2)', 'HTN ADM (F03Q07)', 'HLD (F03Q09) (yes=1, no=2, unknwon =98)', 'STROKE Hx (F03Q01) (defined as prior stroke/TIA Hx)', 'AF (F03Q04)', 'CAD (F3Q06)', 'SMOKING (F03Q13)  (defined as current=1,former=2,never=3,unknown=4)', 'PLT  (x 10^3 / mm^3) (F15Q07)  Use score prior to randomization denoted as " - "', 'INR (F15Q09)  Use score prior to randomization denoted as " - "', 'APTT (sec) (F15Q08)  Use score prior to randomization denoted as " - "', 'EVD F07Q05 (called intraventricular catheter received)', 'HCM F07Q08 (defined as immediate hematoma evac)', 'ICUCARE, F1Q01 represent Days in ICU --> Made into  1=Yes and 0= no', 'INTUBATED (used  F08Q04 ; Not F07Q02)', 'Site of Bleed (F2Q07)  1=BG, 2=thalamus, 3=lobar) (pontine / cerebellar excluded) (Use variable 45 in Qatar: 1=BG, 2=thalamus, use 3 & 6 to count as lobar, exclude pts with 4,5,7)', 'GCSADM F33Q04', 'VOLUME 1 ( on admission) in mm3 -CTCR Q 12', 'IVH (F33QO3) (1 = present , 0==absent)', 'HEM STATUS ON REPEAT CT (0=no change, 1=increased size, 2= decreased size) --> calculate from volume 1 and 2 and change into cm3', 'MORT 90 - (F20Q04 MIN)', 'MRS90 (0 - mRS 0-3 , 1 - mRS 4-6)  (Form 21 primary outcome )', 'AGE > 80   (ask Statistician for help)', 'oICH_score', 'mICH_score', 'ICH_GS_score', 'LSICH_score', 'ICH_FOS_score', 'Max_ICH_score'
])
unmapped_columns = dataset2_columns - set(column_mapping.values())

# Display the mapping and unmapped columns
print("Column Mapping from Dataset 1 to Dataset 2:")
for key, value in column_mapping.items():
    print(f"{key}: {value}")

print("\nColumns in Dataset 2 that did not map to Dataset 1:")
for column in sorted(unmapped_columns):
    print(column)

import pandas as pd

dictionary = {
    'abstract_algebra': 'STEM',
    'anatomy': 'STEM',
    'astronomy': 'STEM',
    'business_ethics': 'Other',
    'clinical_knowledge': 'Other',
    'college_biology': 'STEM',
    'college_chemistry': 'STEM',
    'college_computer_science': 'STEM',
    'college_mathematics': 'STEM',
    'college_medicine': 'Other',
    'college_physics': 'STEM',
    'computer_security': 'STEM',
    'conceptual_physics': 'STEM',
    'econometrics': 'Social Sciences',
    'electrical_engineering': 'STEM',
    'elementary_mathematics': 'STEM',
    'formal_logic': 'Humanities',
    'global_facts': 'Other',
    'high_school_biology': 'STEM',
    'high_school_chemistry': 'STEM',
    'high_school_computer_science': 'STEM',
    'high_school_european_history': 'Humanities',
    'high_school_geography': 'Social Sciences',
    'high_school_government_and_politics': 'Social Sciences',
    'high_school_macroeconomics': 'Social Sciences',
    'high_school_mathematics': 'STEM',
    'high_school_microeconomics': 'Social Sciences',
    'high_school_physics': 'STEM',
    'high_school_psychology': 'Social Sciences',
    'high_school_statistics': 'STEM',
    'high_school_us_history': 'Humanities',
    'high_school_world_history': 'Humanities',
    'human_aging': 'Other',
    'human_sexuality': 'Social Sciences',
    'international_law': 'Humanities',
    'jurisprudence': 'Humanities',
    'logical_fallacies': 'Humanities',
    'machine_learning': 'STEM',
    'management': 'Other',
    'marketing': 'Other',
    'medical_genetics': 'Other',
    'miscellaneous': 'Other',
    'moral_disputes': 'Humanities',
    'moral_scenarios': 'Humanities',
    'nutrition': 'Other',
    'philosophy': 'Humanities',
    'prehistory': 'Humanities',
    'professional_accounting': 'Other',
    'professional_law': 'Humanities',
    'professional_medicine': 'Other',
    'professional_psychology': 'Social Sciences',
    'public_relations': 'Social Sciences',
    'security_studies': 'Social Sciences',
    'sociology': 'Social Sciences',
    'us_foreign_policy': 'Social Sciences',
    'virology': 'Other',
    'world_religions': 'Humanities'
}

data_path = 'xxx/output/mmlu_experiment/intermediate_data.json'

df = pd.read_json(data_path)
df['subject'] = df['metadata'].apply(lambda x: dictionary[x['subject']])
df['em'] = df['output'].apply(lambda x: x['metric_score']['em'])
subject_em_avg = df.groupby('subject')['em'].mean()
print(subject_em_avg)
# print subject count
# subject_count = df.groupby('subject').size()
# print(subject_count)
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt

def carregar_dados(nomeArquivo):
    print("\n\n## CARREGANDO OS DADOS...")
    dados = None
    try:
        dados = pd.read_csv(nomeArquivo, sep=",", encoding="utf8")
    except:
        print("Erro na leitura dos dados! NÃO ENTRE EM PÂNICO e verifique os dados!")
    return dados

def preparar_dados(dados):
    print("\n\n## PREPARANDO OS DADOS...")
    print("\nVISÃO GERAL DOS DADOS: ")
    print(dados.info())

    print("\nVISUALIZANDO OS 5 PRIMEIROS REGISTROS: ")
    print(dados.head())

    print("\nVISUALIZANDO  OS 5 ÚLTIMOS REGISTROS: ")
    print(dados.tail())

    print("\nDESCRIÇÃO ESTATÍSTICA DOS DADOS: ")
    print(dados.describe())

    print("\nDESCRIÇÃO ESTATÍSTICA DOS DADOS: ")
    print('\ncolumn 2 - age | IDADE')
    print(dados['age'].describe())

    print('\ncolumn 4 - study_hours_per_day | HORAS DE ESTUDO POR DIA')
    print(dados['study_hours_per_day'].describe())

    print('\ncolumn 5 - social_media_hours | HORAS EM REDES SOCIAIS')
    print(dados['social_media_hours'].describe())

    print('\ncolumn 6 - netflix_hours | HORAS NO NETFLIX')
    print(dados['netflix_hours'].describe())

    print('\ncolumn 8 - attendance_percentage | PORCENTAGEM DE PRESENÇA')
    print(dados['attendance_percentage'].describe())

    print('\ncolumn 9 - sleep_hours | HORAS DE SONO')
    print(dados['sleep_hours'].describe())

    print('\ncolumn 11 - exercise_frequency | FREQUÊNCIA DE EXERCÍCIO')
    print(dados['exercise_frequency'].describe())

    print('\ncolumn 14 - mental_health_rating | AVALIAÇÃO DA SAÚDE MENTAL')
    print(dados['mental_health_rating'].describe())

    print('\ncolumn 16 - exam_score | NOTA DA PROVA')
    print(dados['exam_score'].describe())
    
    print(f"\nTOTAL DE DADOS DUPLICADOS: {dados.duplicated().sum()}")
    if dados.duplicated().sum() != 0:
        print(dados.duplicated())

    print(f"\nPreenchendo valores da coluna 12 (parental_education_level) de None para Untold")
    dados['parental_education_level'] = dados['parental_education_level'].fillna("Untold")
    
    print(f"\nVALORES DO DATAFRAME: {dados.values}")
    print(f"\nLINHA x COLUNA: {dados.shape}")
    print(f"\nTOTAL DE VALORES NULOS:\n{dados.isnull().sum()}")
    print(f"\nTIPOS DE DADOS:\n{dados.dtypes}")
    
    print(f"\nGÊNERO: {dados['gender'].unique()}") 
    print(f"\nTRABALHA: {dados['part_time_job'].unique()}")
    print(f"\nDIETA: {dados['diet_quality'].unique()}")
    print(f"\nNÍVEL DE EDUCAÇÃO PARENTAL: {dados['parental_education_level'].unique()}")
    print(f"\nQUALIDADE DA INTERNET: {dados['internet_quality'].unique()}")
    print(f"\nPARTICIPAÇÃO EXTRACURRICULAR: {dados['extracurricular_participation'].unique()}")

    return dados

def transformar_dados(dados):
    print("\n\n## TRANSFORMANDO OS DADOS...")
    dados.drop_duplicates(inplace=True)
    dados.drop(columns=["student_id"], inplace=True)
    dados.dropna(inplace=True)

    encoders = {}
    colunas_categoricas = [
        'gender', 
        'part_time_job', 
        'diet_quality',
        'parental_education_level',
        'internet_quality',
        'extracurricular_participation'
    ]

    for coluna in colunas_categoricas:
        encoders[coluna] = LabelEncoder()
        dados[coluna] = encoders[coluna].fit_transform(dados[coluna])

    print("\nMAPEAMENTO DOS VALORES:")
    for coluna, encoder in encoders.items():
        print(f"\n{coluna}:")
        for valor, codigo in zip(encoder.classes_, range(len(encoder.classes_))):
            print(f"   {valor} → {codigo}")

    print(f"\nTIPOS DE DADOS APÓS TRANSFORMAÇÃO:\n{dados.dtypes}")
    
    return dados, encoders

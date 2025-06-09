import pandas as pd
from sklearn.preprocessing import LabelEncoder 


def carregar_dados(nomeArquivo):
    dados = None
    try:
        dados = pd.read_csv(nomeArquivo, sep=",", encoding="utf8")
    except:
        print("Erro na leitura dos dados! NÃO ENTRE EM PÂNICO e verifique os dados!")
    return dados

def preparar_dados(dados):
    print("\nVISÃO GERAL DOS DADOS: ")
    print(dados.info())

    print("\nVISUALZINDO OS 5 PRIMEIROS REGISTROS: ")
    print(dados.head())

    print("\nVISUALZINDO OS 5 ÚLTIMOS REGISTROS: ")
    print(dados.tail())

    print("\nDESCRIÇÃO ESTATÍSTICA DOS DADOS: ")
    print(dados.describe())
    
    print(f"\nTOTAL DE DADOS DUPLICADOS: {dados.duplicated().sum()}")
    if dados.duplicated().sum() != 0:
        print(dados.duplicated())
    
    print(f"\nVALORES DO DATAFRAME: {dados.values}")
    print(f"\nLINHA x COLUNA: {dados.shape}")
    print(f"\nTOTAL DE VALORES NULOS: {dados.isnull().sum()}")
    print(f"\nTIPOS DE DADOS: {dados.dtypes}")
    
    print(f"\nGÊNERO: {dados['gender'].unique()}") 
    print(f"\nTRABALHA: {dados['part_time_job'].unique()}")
    print(f"\nDIETA: {dados['diet_quality'].unique()}")
    print(f"\nNÍVEL DE EDUCAÇÃO PARENTAL: {dados['parental_education_level'].unique()}")
    print(f"\nQUALIDADE DA INTERNET: {dados['internet_quality'].unique()}")
    print(f"\nPARTICIPAÇÃO EXTRACURRICULAR: {dados['extracurricular_participation'].unique()}")

    return dados

def transformar_dados(dados):
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

    return dados
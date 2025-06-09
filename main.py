import bib_mostly_harmless as mh

NOMEARQUIVO = "data/student_habits_performance.csv"

dados = mh.carregar_dados(NOMEARQUIVO)

if dados is not None:
    dados = mh.preparar_dados(dados)
    dados_transformados = mh.transformar_dados(dados)
    
    print("\nDADOS TRANSFORMADOS COM SUCESSO!")
    print(dados_transformados.head())
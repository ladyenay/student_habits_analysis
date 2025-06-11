import bib.processamento as p
import bib.visualizacao as v

NOMEARQUIVO = "data/student_habits_performance.csv"

dados = p.carregar_dados(NOMEARQUIVO)

if dados is not None:
    dados = p.preparar_dados(dados)
    dados_transformados, encoders = p.transformar_dados(dados)
    
    print("\n\nDADOS TRANSFORMADOS COM SUCESSO!")
    print("\nVISÃO GERAL DOS DADOS: ")
    print(dados.info())

    print("\nVISUALIZANDO OS 5 PRIMEIROS REGISTROS: ")
    print(dados.head())

    print("\nVISUALIZANDO  OS 5 ÚLTIMOS REGISTROS: ")
    print(dados.tail())

    print("\nDESCRIÇÃO ESTATÍSTICA DOS DADOS: ")
    print(dados.describe())
    
    print(f"\nTOTAL DE DADOS DUPLICADOS: {dados.duplicated().sum()}")

    print(f"\nLINHA x COLUNA: {dados.shape}")

    print(f"\nTOTAL DE VALORES NULOS:\n{dados.isnull().sum()}")

    print(f"\nTIPOS DE DADOS:\n{dados.dtypes}")
    

    v.plot_bar_chart(
    dados,
    group_column="gender",
    encoder=encoders["gender"],
    colors=["red", "blue", "purple"],
    title="Estudante x Gênero"
    )
   
    v.plot_histogram(dados, column="sleep_hours", title="Horas de Sono")

    v.plot_pie(
        data=dados,
        column="diet_quality",
        encoder_dict=encoders,
        title="Qualidade da Dieta",
        colors=["blue","green","red"]
    )

    v.plot_scatter(
    dados, 
    "study_hours_per_day", 
    "exam_score",
    title="Notas vs Horas de Estudo"
    )

    v.plot_boxplot(
        data=dados,
        x_col="parental_education_level",
        y_col="exam_score",
        title="Notas por Nível Educacional dos Pais",
        encoder=encoders["parental_education_level"]
    )
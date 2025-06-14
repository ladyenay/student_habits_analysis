
import bib.processamento as p
import bib.visualizacao as v
import pandas as pd

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

    bins = [0, 69, 89, 100] 
    labels = ['baixo', 'medio', 'top']

    dados['grupo_desempenho'] = pd.cut(
        dados['exam_score'],
        bins=bins,
        labels=labels,
        right=False
    )
    comparacao = dados.groupby('grupo_desempenho').agg(
        exam_score_mean=('exam_score', 'mean'),
        students_qtd=('exam_score', 'count'),
        study_hours_per_day_mean=('study_hours_per_day', 'mean'),
        attendance_percentage_mean=('attendance_percentage', 'mean'),
        sleep_hours_mean=('sleep_hours', 'mean'),
        exercise_frequency_mean=('exercise_frequency', 'mean'),
        netflix_hours_mean=('netflix_hours', 'mean'),
        mental_health_rating_median=('mental_health_rating', 'median'),
        social_media_hours_mean=('social_media_hours', 'mean')
    ).round(1)
    print(comparacao)

    v.plot_comparativo_grupos(
        comparacao=comparacao,
        variaveis=['sleep_hours_mean'],
        title="Comparação de Hora de Sono por Grupo de Desempenho",
        palette="mako", 
        rotacao_x=30
    )
    v.plot_bar_chart(
    dados,
    group_column="gender",
    encoder=encoders["gender"],
    colors=["red", "blue", "purple"],
    title="Estudante x Gênero"
    )
   
    # v.plot_histogram(dados, column="exam_score", title="Nota Exame")

    # v.plot_pie(
    #     data=dados,
    #     column="internet_quality",
    #     encoder_dict=encoders,
    #     title="Qualidade da Internet",
    #     colors=["blue","green","red"]
    # )

    # v.plot_scatter(
    # dados, 
    # "exam_score", 
    # "extracurricular_participation",
    # title="Horas de Estudo vs "
    # )

    # v.plot_boxplot(
    #     data=dados,
    #     x_col="extracurricular_participation",
    #     y_col="exam_score",
    #     title="Participação Extracurricular",
    #     encoder=encoders["extracurricular_participation"]
    # )
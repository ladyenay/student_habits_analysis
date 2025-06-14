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

    # gráficos de barras: mude o 'group_column' para a coluna que deseja analisar e modifique o título
    v.plot_bar_chart( 
    dados,
    group_column="age",
    #encoder=encoders[""],  # coloque apenas para os dados categóricos, deixe comentado caso o group_column não seja um dado categórico
    # colors=["red", "blue", "purple"], # define as cores de acordo com a quantidade de colunas, caso contrário as cores serão aleatórias
    title="Estudante x Idade"
    )

    # histogramas: mude 'group_column' para a coluna que deseja analisar e modifique o título
    v.plot_histogram(dados, column="exam_score", title="Nota Exame") 

    # gráficos de pizza: mude o 'group_column' para a coluna que deseja analisar e modifique o título
    v.plot_pie(
        data=dados,
        column="part_time_job",
        encoder_dict=encoders, # coloque apenas para os dados categóricos, deixe comentado caso o group_column não seja um dado categórico
        title="Trabalha Meio Período", 
        colors=["red","blue"] # define as cores de acordo com a quantidade de valores
    )

    # gráficos de correlação: mude o eixos de acordo com as colunas que deseja analisar a correlação e modifique o título
    v.plot_scatter(
        dados, 
        x_col="exam_score", # eixo x
        y_col="extracurricular_participation", # eixo y
        title="Correlacao Atividades Extracurriculares"
    )

    # gráficos de boxplot: mude o eixos de acordo com as colunas que deseja analisar e modifique o título
    v.plot_boxplot(
        data=dados,
        x_col="extracurricular_participation", # eixo x
        y_col="exam_score",                    #eixo y
        title="Participação Extracurricular", # coloque apenas para os dados categóricos, deixe comentado caso o group_column não seja um dado categórico
        encoder=encoders["extracurricular_participation"]
    )

    # Dividindo os estudantes em grupos
    bins = [0, 69, 89, 100] 
    labels = ['baixo', 'medio', 'top'] # Grupos de Desempenho

    dados['grupo_desempenho'] = pd.cut(
        dados['exam_score'],
        bins=bins,
        labels=labels,
        right=False
    ) 
    comparacao = dados.groupby('grupo_desempenho').agg( # agrupando os grupos, para cada varriável, usando a média ou mediana
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
        variaveis=['exam_score_mean'],  # mude aqui a variável que deseja comparar com o Grupo de Desempenho
        title="Comparação Nota por Grupo de Desempenho",
        palette="mako", 
        rotacao_x=30
    )

    media_geral = dados_transformados['exam_score'].mean()

    media_por_age = dados_transformados.groupby('age')['exam_score'].mean().reset_index()
    print("\n--- Médias Calculadas por Idade ---")
    media_por_age.loc[len(media_por_age)] = ['Total', media_geral]
    print(media_por_age)
    print("\n" + "="*30 + "\n")

    media_por_genero = dados_transformados.groupby('gender')['exam_score'].mean().reset_index()
    print("\n--- Médias Calculadas por Gênero ---")
    media_por_genero.loc[len(media_por_genero)] = ['Total', media_geral]
    print(media_por_genero)
    print("\n" + "="*30 + "\n")

    media_por_study_hours_per_day = dados_transformados.groupby('study_hours_per_day')['exam_score'].mean().reset_index()
    print("\n--- Médias Calculadas por Horas de Estudo por Dia ---")
    media_por_study_hours_per_day.loc[len(media_por_study_hours_per_day)] = ['Total', media_geral]
    print(media_por_study_hours_per_day)
    print("\n" + "="*30 + "\n") 

    media_por_social_media_hours = dados_transformados.groupby('social_media_hours')['exam_score'].mean().reset_index()
    print("\n--- Médias Calculadas por Horas em Redes Sociais ---")
    media_por_social_media_hours.loc[len(media_por_social_media_hours)] = ['Total', media_geral]
    print(media_por_social_media_hours)
    print("\n" + "="*30 + "\n")

    media_por_netflix_hours = dados_transformados.groupby('netflix_hours')['exam_score'].mean().reset_index()
    print("\n--- Médias Calculadas por Horas no Netflix ---")
    media_por_netflix_hours.loc[len(media_por_netflix_hours)] = ['Total', media_geral]
    print(media_por_netflix_hours)
    print("\n" + "="*30 + "\n")

    media_por_part_time_job = dados_transformados.groupby('part_time_job')['exam_score'].mean().reset_index()
    print("\n--- Médias Calculadas por Trabalho de Meio Período ---")
    media_por_part_time_job.loc[len(media_por_part_time_job)] = ['Total', media_geral]
    print(media_por_part_time_job)
    print("\n" + "="*30 + "\n")

    media_por_attendance = dados_transformados.groupby('attendance_percentage')['exam_score'].mean().reset_index()
    print("\n--- Médias Calculadas por Frequência de Presença ---")
    media_por_attendance.loc[len(media_por_attendance)] = ['Total', media_geral]
    print(media_por_attendance)
    print("\n" + "="*30 + "\n")

    media_por_sleep_hours = dados_transformados.groupby('sleep_hours')['exam_score'].mean().reset_index()
    print("\n--- Médias Calculadas por Horas de Sono ---")
    media_por_sleep_hours.loc[len(media_por_sleep_hours)] = ['Total', media_geral]
    print(media_por_sleep_hours)
    print("\n" + "="*30 + "\n")

    media_por_diet_quality = dados_transformados.groupby('diet_quality')['exam_score'].mean().reset_index()
    print("\n--- Médias Calculadas por Qualidade da Dieta ---")
    media_por_diet_quality.loc[len(media_por_diet_quality)] = ['Total', media_geral]
    print(media_por_diet_quality)
    print("\n" + "="*30 + "\n")
    
    media_por_exercise_frequency = dados_transformados.groupby('exercise_frequency')['exam_score'].mean().reset_index()
    print("\n--- Médias Calculadas por Frequência de Exercício ---")
    media_por_exercise_frequency.loc[len(media_por_exercise_frequency)] = ['Total', media_geral]
    print(media_por_exercise_frequency)
    print("\n" + "="*30 + "\n")

    media_por_parental_education_level = dados_transformados.groupby('parental_education_level')['exam_score'].mean().reset_index()
    print("\n--- Médias Calculadas por Nível de Educação Parental ---")
    media_por_parental_education_level.loc[len(media_por_parental_education_level)] = ['Total', media_geral]
    print(media_por_parental_education_level)
    print("\n" + "="*30 + "\n")

    media_por_internet_quality = dados_transformados.groupby('internet_quality')['exam_score'].mean().reset_index()
    print("\n--- Médias Calculadas por Qualidade da Internet ---")
    media_por_internet_quality.loc[len(media_por_internet_quality)] = ['Total', media_geral]
    print(media_por_internet_quality)
    print("\n" + "="*30 + "\n")

    media_por_mental_health = dados_transformados.groupby('mental_health_rating')['exam_score'].mean().reset_index()
    print("\n--- Médias Calculadas por Saúde Mental ---")
    media_por_mental_health.loc[len(media_por_mental_health)] = ['Total', media_geral]
    print(media_por_mental_health)
    print("\n" + "="*30 + "\n")

    media_por_extracurricular_participation = dados_transformados.groupby('extracurricular_participation')['exam_score'].mean().reset_index()
    print("\n--- Médias Calculadas por Participação Extracurricular ---")
    media_por_extracurricular_participation.loc[len(media_por_extracurricular_participation)] = ['Total', media_geral]
    print(media_por_extracurricular_participation)
    print("\n" + "="*30 + "\n")

    # Visualização dos dados
    print("\n\n## GERANDO OS GRAFICOS DOS DADOS...")
    v.plot_bar_chart(
    dados,
    group_column="gender",
    encoder=encoders["gender"],
    colors=["red", "blue", "purple"],
    title="Estudante x Gênero"
    )
   
    v.plotar_grafico_barras_linha_genero(media_por_genero, media_geral)

    v.plotar_grafico_linha_frequencia(media_por_attendance, media_geral)

    v.plotar_grafico_linhas_study_hours(media_por_study_hours_per_day, media_geral)
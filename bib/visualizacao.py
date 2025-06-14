import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import seaborn as sns
import statsmodels.api as sm
from typing import Optional, Dict

def plot_scatter(
    dados: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = None,
    title: str = "",
    palette: Optional[Dict] = None,
    jitter: bool = True,
    alpha: float = 0.7,
    show_regression: bool = True,
    reg_color: Optional[str] = None, 
) -> None:

    plt.figure(figsize=(10, 6))
    if jitter and dados[x_col].nunique() < 10:
        x = dados[x_col] + np.random.normal(0, 0.05, len(dados))
    else:
        x = dados[x_col]
    scatter = sns.scatterplot(
        x=x,
        y=dados[y_col],
        hue=dados[hue_col] if hue_col else None,
        palette=palette,
        alpha=alpha,
    )

    if show_regression:
        if hue_col:
            for grupo in dados[hue_col].unique():
                subset = dados[dados[hue_col] == grupo]
                X = sm.add_constant(subset[x_col])
                modelo = sm.OLS(subset[y_col], X).fit()
                sns.regplot(
                    x=subset[x_col],
                    y=subset[y_col],
                    scatter=False,
                    label=f"{grupo} (R²={modelo.rsquared:.2f})",
                    color=None, 
                )
        else:
            X = sm.add_constant(dados[x_col])
            modelo = sm.OLS(dados[y_col], X).fit()
            sns.regplot(
                x=x,
                y=dados[y_col],
                scatter=False,
                color=reg_color or "red",
                label=f"Global (R²={modelo.rsquared:.2f})",
            )
        plt.legend(loc="best") 

    if hue_col and pd.api.types.is_numeric_dtype(dados[hue_col]):
        handles, labels = scatter.get_legend_handles_labels()
        scatter.legend(
            handles=handles,
            title=hue_col,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
        )

    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_boxplot(
    data,
    x_col: str, 
    y_col: str, 
    title: str = "",
    showfliers: bool = True,
    encoder=None,
):
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=data,
        x=x_col,
        y=y_col,
        showfliers=showfliers
    )
    
    if encoder is not None:
        current_labels = plt.xticks()[0]
        original_labels = [encoder.inverse_transform([code])[0] for code in current_labels]
        plt.xticks(current_labels, original_labels)
    
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_bar_chart(
    data,
    group_column,
    encoder=None,
    colors=None,
    title=None,
    xlabel=None,
    ylabel="Contagem",
    legend=True,
    rotate_labels=True,
    show_values=True,  
    value_format="{:.0f}",  
    ax=None,
    **kwargs
):

    grouped = data.groupby(group_column)
    numeric_labels = list(grouped.groups.keys())
    values = [len(group) for group in grouped.groups.values()]

    if encoder is not None:
        labels = [encoder.inverse_transform([code])[0] for code in numeric_labels]
    else:
        labels = [str(code) for code in numeric_labels]

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    if ax is None:
        fig, ax = plt.subplots()

    bars = ax.bar(labels, values, color=colors, **kwargs)

    if show_values:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., 
                height / 2,  
                value_format.format(height), 
                ha='center', 
                va='center', 
                color='white', 
                fontweight='bold'
            )

    if legend:
        ax.legend(bars, labels, title=group_column)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rotate_labels:
        ax.xaxis.set_major_locator(FixedLocator(range(len(labels))))
        ax.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()
    return ax

def plot_histogram(data, column=None, title="Histograma", bins=10, color='skyblue'):

    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("Para DataFrames, especifique a coluna com o parâmetro 'column'")
        values = data[column]
    else:
        values = data
    
    plt.figure()
    values.hist(bins=bins, color=color, edgecolor='white')
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def plot_pie(data, column, encoder_dict=None, title="Gráfico de Pizza", colors=None):

    counts = data[column].value_counts()

    if encoder_dict and column in encoder_dict:
        encoder = encoder_dict[column]
        labels = [encoder.inverse_transform([int(idx)])[0] for idx in counts.index]
    else:
        labels = counts.index.tolist()
    
    plt.figure(figsize=(8, 6))
    plt.pie(
        counts.values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white'}
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_comparativo_grupos(
    comparacao: pd.DataFrame,
    variaveis: list,
    title: str = "Comparação entre Grupos de Desempenho",
    figsize: tuple = (12, 6),
    palette: str = "viridis",
    rotacao_x: int = 45,
    decimal_places: int = 1,
):

    plt.figure(figsize=figsize)
    
    dados_plot = comparacao[variaveis].reset_index()
    
    dados_plot.columns = [col.replace("_mean", "").replace("_median", "") for col in dados_plot.columns]

    dados_plot = dados_plot.melt(
        id_vars="grupo_desempenho",
        var_name="variavel",
        value_name="valor"
    )
    dados_plot = dados_plot[~dados_plot["variavel"].astype(str).str.contains("0.0")] 

    ax = sns.barplot(
        data=dados_plot,
        x="variavel",
        y="valor",
        hue="grupo_desempenho",
        palette=palette,
    )

    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.annotate(
                f"{height:.{decimal_places}f}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center',
                va='center',
                xytext=(0, 5),
                textcoords='offset points',
                fontsize=9,
                color='black'
            )
    
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel("Valor Médio/Mediano")
    plt.xticks(rotation=rotacao_x, ha="right")
    plt.legend(title="Grupo de Desempenho", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    ax.set_xticklabels([label.get_text().replace("0.0", "").strip() for label in ax.get_xticklabels()])
    
    plt.tight_layout()
    plt.show()

def plotar_grafico_barras_linha_genero(media_por_genero, media_geral):
    plt.figure(figsize=(10, 6))

    bar_labels = ['Mulher', 'Homem', 'Indefinido']
    bar_values = media_por_genero['exam_score'].tolist()[:-1] 
    bars = plt.bar(bar_labels, bar_values, color=['skyblue', 'lightcoral', 'lightgreen'])

    plt.axhline(y=media_geral, color='purple', linestyle='--', linewidth=2, label=f'Média Geral: {media_geral:.2f}')
    plt.xlabel('Gênero', fontsize=12)
    plt.ylabel('Média da Nota', fontsize=12)
    plt.title('Média de Notas por Gênero e Média Geral', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom', fontsize=10)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plotar_grafico_linha_frequencia(media_por_attendance, media_geral):
    plt.figure(figsize=(10, 6))
    line_labels = media_por_attendance['attendance_percentage'].tolist()[:-1]
    line_values = media_por_attendance['exam_score'].tolist()[:-1]  
    plt.plot(line_labels, line_values, marker='o', color='blue', label='Média por Frequência')
    plt.axhline(y=media_geral, color='purple', linestyle='--', linewidth=2, label=f'Média Geral: {media_geral:.2f}')
    plt.xlabel('Frequência de Presença (%)', fontsize=12)
    plt.ylabel('Média da Nota', fontsize=12)
    plt.title('Média de Notas por Frequência de Presença e Média Geral', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plotar_grafico_linhas_study_hours(media_por_study_hours_per_day, media_geral):
    plt.figure(figsize=(10, 6))
    line_labels = media_por_study_hours_per_day['study_hours_per_day'].tolist()[:-1]
    line_values = media_por_study_hours_per_day['exam_score'].tolist()[:-1]
    plt.plot(line_labels, line_values, marker='o', color='blue', label='Média por Horas de Estudo')
    plt.axhline(y=media_geral, color='purple', linestyle='--', linewidth=2, label=f'Média Geral: {media_geral:.2f}')
    plt.xlabel('Horas de Estudo por Dia', fontsize=12)
    plt.ylabel('Média da Nota', fontsize=12)
    plt.title('Média de Notas por Horas de Estudo por Dia e Média Geral', fontsize=14, fontweight='bold')
    plt.ylim(10, 110)

    min_x = int(min(line_labels))
    max_x = int(max(line_labels))
    plt.xticks(range(min_x, max_x + 1), rotation=45)  
    # plt.xticks()
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
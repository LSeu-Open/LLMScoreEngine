# ------------------------------------------------------------------------------------------------
# License
# ------------------------------------------------------------------------------------------------

# Copyright (c) 2025 LSeu-Open
# 
# This code is licensed under the MIT License.
# See LICENSE file in the root directory

# ------------------------------------------------------------------------------------------------
# Description
# ------------------------------------------------------------------------------------------------

"""
This module generates a comprehensive HTML report with plots from the most recent CSV data.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

RESULTS_DIR = 'Results'
REPORTS_DIR = os.path.join(RESULTS_DIR, 'Reports')

# ------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------

def find_latest_csv_report():
    """
    Finds the most recent CSV report file in the Results/Reports directory.
    
    Returns:
        str: The full path to the latest CSV report, or None if not found.
    """
    if not os.path.exists(REPORTS_DIR):
        print(f"Error: Reports directory '{REPORTS_DIR}' not found.")
        return None
        
    csv_files = [f for f in os.listdir(REPORTS_DIR) if f.startswith('LLM-Scoring-Engine_report_') and f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV report files found in the {REPORTS_DIR} directory.")
        return None

    latest_file = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(REPORTS_DIR, f)))
    return os.path.join(REPORTS_DIR, latest_file)

def read_report_data(file_path):
    """
    Reads data from a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: A DataFrame containing the report data, or None on error.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file {os.path.basename(file_path)}: {e}")
        return None

def generate_executive_summary(df):
    """
    Generates insights for the executive summary.
    
    Args:
        df (pd.DataFrame): The report data.
        
    Returns:
        dict: A dictionary containing summary data points.
    """
    summary_data = {}
    
    # Highest final score
    top_model_by_score = df.loc[df['final_score'].idxmax()]
    summary_data['top_model_by_score'] = f"{top_model_by_score['model_name']} (Score: {top_model_by_score['final_score']:.2f})"

    # Best performance-to-cost ratio
    if 'total_price' in df.columns and df['total_price'].gt(0).any():
        df_cost_eff = df[df['total_price'] > 0].copy()
        df_cost_eff['cost_efficiency'] = df_cost_eff['final_score'] / df_cost_eff['total_price']
        top_model_by_cost = df_cost_eff.loc[df_cost_eff['cost_efficiency'].idxmax()]
        summary_data['top_model_by_cost'] = f"{top_model_by_cost['model_name']} (Score/Price: {top_model_by_cost['cost_efficiency']:.2f})"
    
    # Architecture with the highest median score
    if 'architecture' in df.columns and not df['architecture'].isnull().all():
        median_scores = df.groupby('architecture')['final_score'].median()
        top_arch = median_scores.idxmax()
        summary_data['top_arch'] = f"{top_arch} (Median Score: {median_scores.max():.2f})"
        
    return summary_data

def generate_scatter_plots(df):
    """
    Generates scatter plots of parameter count vs. various scores.
    
    Args:
        df (pd.DataFrame): The report data.
        
    Returns:
        go.Figure: A Plotly figure object.
    """
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Parameter Count vs. Final Score",
            "Parameter Count vs. Entity Score",
            "Parameter Count vs. Dev Score",
            "Parameter Count vs. Technical Score",
            "Parameter Count vs. Community Score"
        ),
        vertical_spacing=0.15
    )
    fig.add_trace(go.Scatter(x=df['param_count'], y=df['final_score'], mode='markers', name='Final Score', hovertext=df['model_name']), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['param_count'], y=df['entity_score'], mode='markers', name='Entity Score', hovertext=df['model_name']), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['param_count'], y=df['dev_score'], mode='markers', name='Dev Score', hovertext=df['model_name']), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['param_count'], y=df['technical_score'], mode='markers', name='Technical Score', hovertext=df['model_name']), row=2, col=2)
    fig.add_trace(go.Scatter(x=df['param_count'], y=df['community_score'], mode='markers', name='Community Score', hovertext=df['model_name']), row=3, col=1)
    fig.update_layout(title_text="Model Performance vs. Parameter Count", height=1200, showlegend=False)
    fig.update_xaxes(type="log", title_text="Parameter Count (log scale)")
    fig.update_yaxes(title_text="Score")
    return fig

def generate_score_composition_chart(df):
    """
    Generates a faceted bar chart of score composition by parameter count.
    
    Args:
        df (pd.DataFrame): The report data.
        
    Returns:
        go.Figure: A Plotly figure object.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Parameters > 70B", "Parameters: 15B - 70B", "Parameters: 7B - 14B", "Parameters < 7B"),
        vertical_spacing=0.3
    )
    
    bins = [
        {'query': 'param_count > 70', 'pos': (1, 1)},
        {'query': '15 <= param_count <= 70', 'pos': (1, 2)},
        {'query': '7 <= param_count < 15', 'pos': (2, 1)},
        {'query': 'param_count < 7', 'pos': (2, 2)}
    ]
    
    score_components = ['entity_score', 'dev_score', 'community_score', 'technical_score']
    
    score_colors = {
        'entity_score': '#636EFA',
        'dev_score': '#EF553B',
        'community_score': '#00CC96',
        'technical_score': '#AB63FA'
    }

    show_legend = True
    
    for b in bins:
        df_bin = df.query(b['query']).sort_values('final_score', ascending=False)
        if not df_bin.empty:
            for score in score_components:
                fig.add_trace(
                    go.Bar(
                        x=df_bin['model_name'], 
                        y=df_bin[score], 
                        name=score,
                        legendgroup=score,
                        showlegend=show_legend,
                        marker_color=score_colors.get(score)
                    ),
                    row=b['pos'][0], col=b['pos'][1]
                )
            show_legend = False
    
    fig.update_layout(
        title_text="Score Composition by Parameter Count (sorted by Final Score)",
        barmode='stack',
        height=1300,
        legend_title_text='Score Components'
    )
    fig.update_xaxes(tickangle=-60)
    return fig

def generate_average_score_composition_pie_charts(df):
    """
    Generates pie charts showing the average score composition by parameter count.
    
    Args:
        df (pd.DataFrame): The report data.
        
    Returns:
        go.Figure: A Plotly figure object.
    """
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]],
        subplot_titles=("Parameters > 70B", "Parameters: 15B - 70B", "7B - 14B", "< 7B")
    )

    bins = [
        {'query': 'param_count > 70'},
        {'query': '15 <= param_count <= 70'},
        {'query': '7 <= param_count < 15'},
        {'query': 'param_count < 7'}
    ]
    
    score_components = ['entity_score', 'dev_score', 'community_score', 'technical_score']
    
    score_colors = {
        'entity_score': '#636EFA',
        'dev_score': '#EF553B',
        'community_score': '#00CC96',
        'technical_score': '#AB63FA'
    }

    for i, b in enumerate(bins):
        row, col = (i // 2) + 1, (i % 2) + 1
        df_bin = df.query(b['query'])
        if not df_bin.empty:
            avg_scores = df_bin[score_components].mean()
            fig.add_trace(
                go.Pie(
                    labels=avg_scores.index,
                    values=avg_scores.values,
                    name=b['query'], # Add name for hover info
                    marker_colors=[score_colors.get(s) for s in avg_scores.index]
                ),
                row=row, col=col
            )
    
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    fig.update_layout(
        title_text="Average Score Composition by Parameter Count",
        height=700,
        legend_title_text='Score Components'
    )
    return fig

def generate_performance_vs_cost_chart(df):
    """
    Generates scatter plots for performance vs. cost analysis, highlighting the Pareto frontier.
    
    Args:
        df (pd.DataFrame): The report data.
        
    Returns:
        go.Figure: A Plotly figure object, or None if price data is unavailable.
    """
    if 'total_price' not in df.columns or df['total_price'].isnull().all():
        print("Warning: 'total_price' column not found or is empty. Skipping 'Performance vs. Cost Analysis' chart.")
        return None
        
    df_cost = df.dropna(subset=['total_price', 'final_score', 'entity_score', 'dev_score', 'technical_score']).copy()
    df_cost = df_cost[df_cost['total_price'] > 0]

    if df_cost.empty:
        return None

    df_cost_sorted = df_cost.sort_values(by=['total_price', 'final_score'], ascending=[True, False])
    pareto_frontier = []
    max_score = -1
    for index, row in df_cost_sorted.iterrows():
        if row['final_score'] > max_score:
            pareto_frontier.append(index)
            max_score = row['final_score']
    
    df_cost['is_efficient'] = df_cost.index.isin(pareto_frontier)
    
    df_efficient = df_cost[df_cost['is_efficient']]
    df_other = df_cost[~df_cost['is_efficient']]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Total Price vs. Final Score", "Total Price vs. Entity Score",
            "Total Price vs. Dev Score", "Total Price vs. Technical Score"
        )
    )

    # Add traces for non-efficient and efficient models
    scores_to_plot = ['final_score', 'entity_score', 'dev_score', 'technical_score']
    for i, score in enumerate(scores_to_plot):
        row, col = (i // 2) + 1, (i % 2) + 1
        show_legend_flag = (i == 0)
        
        # Other models
        fig.add_trace(go.Scatter(
            x=df_other['total_price'], y=df_other[score], mode='markers', 
            name='Other Models', hovertext=df_other['model_name'],
            marker={'color': 'lightgray'}, showlegend=show_legend_flag, legendgroup='group1'
        ), row=row, col=col)
        
        # Efficient models
        fig.add_trace(go.Scatter(
            x=df_efficient['total_price'], y=df_efficient[score], mode='markers', 
            name='Efficient Frontier', hovertext=df_efficient['model_name'],
            marker={'symbol': 'star', 'size': 10, 'color': '#EF553B'}, showlegend=show_legend_flag, legendgroup='group2'
        ), row=row, col=col)

    fig.update_layout(
        title_text='Performance vs. Cost Analysis (Log Scale)',
        yaxis_title="Score",
        height=800,
        legend_title_text='Model Type'
    )
    fig.update_xaxes(type="log", title_text="Total Price (Input + Output USD/Mtok, log scale)")
    return fig

def generate_architecture_performance_violin_plot(df):
    """
    Generates a violin plot of final score distribution by architecture.
    
    Args:
        df (pd.DataFrame): The report data.
        
    Returns:
        go.Figure: A Plotly figure object, or None if architecture data is unavailable.
    """
    df_arch = df.dropna(subset=['architecture', 'final_score']).copy()
    if df_arch.empty:
        return None
        
    arch_counts = df_arch['architecture'].value_counts()
    df_arch['architecture_with_count'] = df_arch['architecture'].apply(lambda x: f"{x} (N={arch_counts[x]})")
    fig = go.Figure()
    fig.add_trace(go.Violin(
        x=df_arch['architecture_with_count'], y=df_arch['final_score'], 
        name='Final Score', box_visible=True, points='all'
    ))
    fig.update_layout(
        title_text="Final Score Distribution by Architecture",
        xaxis_title="Architecture", yaxis_title="Final Score"
    )
    return fig

def generate_radar_chart(df, model_names):
    """
    Generates a radar chart for comparing multiple models across score categories.
    
    Args:
        df (pd.DataFrame): The full report data.
        model_names (list): A list of model names to plot.
        
    Returns:
        go.Figure: A Plotly figure object.
    """
    categories = ['entity_score', 'dev_score', 'community_score', 'technical_score']
    
    fig = go.Figure()

    for model_name in model_names:
        model_data = df[df['model_name'] == model_name]
        if not model_data.empty:
            values = model_data[categories].values.flatten().tolist()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=model_name
            ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 30]  # Max score for entity/dev is 30, a good upper bound
            )),
        showlegend=True,
        title='Model Score Composition Comparison'
    )
    return fig

def render_html_report(context, template_dir=None):
    """
    Renders the HTML report using a Jinja2 template and saves it to a file.
    
    Args:
        context (dict): A dictionary of variables to pass to the template.
        template_dir (str, optional): The directory where the template is located. 
                                     Defaults to the directory of this script.
    """
    search_path = template_dir if template_dir is not None else os.path.dirname(__file__)
    env = Environment(loader=FileSystemLoader(search_path))
    template = env.get_template('report_template.html')
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    output_file = os.path.join(REPORTS_DIR, 'model_performance_report.html')
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(template.render(context))
        print(f"Report successfully generated and saved to {output_file}")
    except Exception as e:
        print(f"Error writing HTML report to {output_file}: {e}")

def generate_report(template_dir=None):
    """
    Generates a comprehensive HTML report with plots from the most recent CSV data.
    
    Args:
        template_dir (str, optional): The directory containing the 'report_template.html'.
                                     Used for testing purposes.
    """
    latest_csv_path = find_latest_csv_report()
    if not latest_csv_path:
        return

    df = read_report_data(latest_csv_path)
    if df is None:
        return

    # Create a total_price column for cost analysis
    if 'input_price' in df.columns and 'output_price' in df.columns:
        df['total_price'] = df['input_price'].fillna(0) + df['output_price'].fillna(0)

    # Prepare data for leaderboard
    df_leaderboard = df.sort_values(by='final_score', ascending=False).copy()
    # Ensure all required columns are present, fill missing with empty string
    all_cols = ['model_name', 'final_score', 'param_count', 'architecture', 'input_price', 'output_price', 'entity_score', 'dev_score', 'community_score', 'technical_score']
    for col in all_cols:
        if col not in df_leaderboard.columns:
            df_leaderboard[col] = ''
    df_leaderboard = df_leaderboard[all_cols] # Ensure consistent column order

    df_leaderboard.fillna('', inplace=True)
    leaderboard_data = df_leaderboard.to_dict(orient='records')
    leaderboard_headers = list(df_leaderboard.columns)
    final_score_col_index = leaderboard_headers.index('final_score') if 'final_score' in leaderboard_headers else -1
    default_visible_cols = ['model_name', 'final_score', 'architecture', 'param_count']
    
    # Get min/max for param_count slider
    param_counts = pd.to_numeric(df['param_count'], errors='coerce').dropna()
    min_param = int(param_counts.min()) if not param_counts.empty else 0
    max_param = int(param_counts.max()) if not param_counts.empty else 1000

    # Prepare cost-efficiency data
    cost_efficiency_data = None
    if 'total_price' in df.columns and df['total_price'].gt(0).any():
        df_cost = df[df['total_price'] > 0].copy()
        df_cost['cost_efficiency'] = (df_cost['final_score'] / df_cost['total_price']).round(2)
        cost_efficiency_data = df_cost.sort_values(by='cost_efficiency', ascending=False)
        cost_efficiency_data = cost_efficiency_data[['model_name', 'cost_efficiency', 'final_score', 'input_price', 'output_price']]
        cost_efficiency_data = cost_efficiency_data.to_dict(orient='records')

    # Generate all components for the report
    summary_data = generate_executive_summary(df)
    fig_scatter = generate_scatter_plots(df)
    fig_bar_faceted = generate_score_composition_chart(df)
    fig_pie_composition = generate_average_score_composition_pie_charts(df)
    fig_cost = generate_performance_vs_cost_chart(df)
    fig_violin = generate_architecture_performance_violin_plot(df)

    # Generate radar chart for the top model by default
    top_model_name = df.loc[df['final_score'].idxmax()]['model_name']
    fig_radar = generate_radar_chart(df, [top_model_name])

    # Prepare template variables
    template_vars = {
        "report_title": "LLM Scoring Engine - Performance Report",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "intro_text": "This report provides a comprehensive analysis of Large Language Model (LLM) performance based on various scoring metrics. Use the tabs below to navigate to different sections of the report. For more details on the scoring metrics, please refer to the <a href='https://github.com/LSeu-Open/AIEnhancedWork/blob/main/Scoring/scoring_framework.md'>scoring metrics documentation</a>.",
        "summary_data": summary_data,
        "leaderboard_data": leaderboard_data,
        "leaderboard_headers": leaderboard_headers,
        "default_visible_cols": default_visible_cols,
        "final_score_col_index": final_score_col_index,
        "min_param": min_param,
        "max_param": max_param,
        "fig_scatter_html": to_html(fig_scatter, full_html=False, include_plotlyjs='cdn'),
        "fig_bar_faceted_html": to_html(fig_bar_faceted, full_html=False, include_plotlyjs=False),
        "fig_pie_composition_html": to_html(fig_pie_composition, full_html=False, include_plotlyjs=False),
        "fig_cost_html": to_html(fig_cost, full_html=False, include_plotlyjs=False) if fig_cost else None,
        "fig_violin_html": to_html(fig_violin, full_html=False, include_plotlyjs=False) if fig_violin else None,
        "fig_radar_html": to_html(fig_radar, full_html=False, include_plotlyjs=False),
        "radar_chart_data": leaderboard_data,
        "cost_efficiency_data": cost_efficiency_data,
        "fig_cost": fig_cost is not None
    }
    
    # Render and save the final HTML report
    render_html_report(template_vars, template_dir=template_dir)

# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    generate_report() 
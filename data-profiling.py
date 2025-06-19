import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import chardet
import duckdb
from ydata_profiling import ProfileReport
from openai import OpenAI
from dotenv import load_dotenv
import os
import tempfile
import base64


def duckdb_profiling(df, file_extension):
    """Gera relatório de profiling usando DuckDB para qualquer formato"""
    conn = duckdb.connect()
    
    # Cria tabela temporária a partir do DataFrame pandas
    conn.register('df_temp', df)
    
    # Coleta metadados básicos
    metadata = conn.execute("""
        SELECT 
            count(*) AS total_linhas,
            (SELECT count(*) FROM information_schema.columns WHERE table_name = 'df_temp') AS total_colunas
        FROM df_temp
    """).fetchdf()
    
    # Coleta estatísticas por coluna (com tratamento para nomes especiais)
    col_stats = []
    columns = conn.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'df_temp'
    """).fetchall()
    
    for col in columns:
        col_name = col[0]
        try:
            # Usa aspas duplas para nomes de colunas com caracteres especiais
            safe_col_name = f'"{col_name}"'
            
            stats = conn.execute(f"""
                SELECT 
                    approx_count_distinct({safe_col_name})::INT AS distinct_count,
                    count({safe_col_name}) AS non_missing,
                    (1 - (count({safe_col_name}) / (SELECT count(*) FROM df_temp)) AS missing_pct,
                    min({safe_col_name}) AS min_val,
                    max({safe_col_name}) AS max_val,
                    avg(try_cast({safe_col_name} AS DOUBLE)) AS mean_val,
                    data_type AS col_type
                FROM df_temp
                CROSS JOIN (SELECT data_type FROM information_schema.columns 
                           WHERE table_name = 'df_temp' AND column_name = '{col_name}')
            """).fetchdf()
            
            col_stats.append({
                'coluna': col_name,
                'tipo': stats['col_type'][0],
                **stats.iloc[0].to_dict()
            })
        except Exception as e:
            st.warning(f"Erro ao analisar coluna {col_name}: {str(e)}")
            continue
    
    return metadata, pd.DataFrame(col_stats)

def plot_column_stats(col_stats):
    """Gera visualizações interativas para as colunas"""
    for _, row in col_stats.iterrows():
        with st.expander(f"**{row['coluna']}** ({row['tipo']})"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Valores Distintos", row['distinct_count'])
                st.metric("Valores Não Nulos", row['non_missing'])
                st.progress(1 - row['missing_pct'], text=f"Completude: {(1 - row['missing_pct'])*100:.1f}%")
            
            with col2:
                if pd.api.types.is_numeric_dtype(row['tipo']):
                    st.metric("Valor Mínimo", f"{row['min_val']:.2f}")
                    st.metric("Valor Máximo", f"{row['max_val']:.2f}")
                    st.metric("Média", f"{row['mean_val']:.2f}")

# Configuração inicial
st.set_page_config(page_title="Profiling de Dados IA", layout="wide")
st.title("📊 Agente de Profiling de Dados")
st.markdown("""
<span style="font-size: 16px; color: #555;">
    Ferramenta inteligente para análise automática da qualidade dos seus dados. Detecte
    problemas, padrões e anomalias, receba recomendações e compreenda suas bases de forma prática, visual e intuitiva.
</span>
""", unsafe_allow_html=True)

# CSS GLOBAL
st.markdown("""
<style>
    /* Importa fonte do Font Awesome */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
            
    /* Fundo branco */
    .stApp {
        background-color: white;
    }
            
    /* Barras de progresso mais largas e azuis */
    .stProgress > div > div > div {
        background-color: #f1eef6 !important;
        height: 16px !important;
    }
            
    /* Texto geral */
    .stApp, .st-bb, .st-at, .st-ae, .st-af, .st-ag, .stMarkdown, .stAlert, 
    .stProgress, .stMetric, .st-expander, .stTextInput, .stNumberInput,
    .stSelectbox, .stSlider, .stDataFrame, .stTable {
        color: black !important;
    }
    
    /* Títulos */
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: black !important;
    }
    
    /* Links */
    a {
        color: #1f77b4 !important;
    }
</style>
""", unsafe_allow_html=True)

# Carrega chave OpenAI do .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Funções auxiliares
def detectar_encoding(arquivo):
    """Detecta o encoding do arquivo usando chardet"""
    rawdata = arquivo.read()
    resultado = chardet.detect(rawdata)
    arquivo.seek(0)  # Volta ao início do arquivo
    return resultado['encoding']

def detectar_separador(arquivo, encoding):
    """Detecta o separador do CSV analisando as primeiras linhas"""
    conteudo = arquivo.read().decode(encoding)
    arquivo.seek(0)
    
    # Testa os separadores mais comuns
    for sep in [',', ';', '\t', '|']:
        if sep in conteudo.split('\n')[0]:
            return sep
    return ','  # Fallback padrão

def calcular_scores(df):
    """
    Calcula scores de qualidade de dados (1-5) para um DataFrame com base em 5 critérios:
    1. Completude (valores não nulos)
    2. Unicidade (registros únicos)
    3. Consistência (tipos de dados)
    4. Precisão (outliers numéricos)
    5. Integridade (valores semânticos e formatos)
    """
    scores = {}
    diagnostico_colunas = {
            "Completude": [],
            "Unicidade": [],
            "Consistência": [],
            "Precisão": [],
            "Integridade": [],
        }
    # 1. COMPLETUDE - Verifica valores não nulos
    null_pct = df.isnull().mean()
    col_completude_ruim = null_pct[null_pct > 0.3].index.tolist()
    completude_score = (1 - null_pct.mean()) * 100
    scores["Completude"] = max(1, round(completude_score / 20))
    diagnostico_colunas["Completude"] = col_completude_ruim

    # 2. UNICIDADE - Verifica registros duplicados
    if df.duplicated().mean() > 0.2:
        diagnostico_colunas["Unicidade"] = ["⚠️ Muitos registros duplicados"]
    unicidade_score = (1 - df.duplicated().mean()) * 100
    scores["Unicidade"] = max(1, round(unicidade_score / 20))

    # 3. CONSISTÊNCIA - Garante que o conteúdo bate com o dtype declarado da coluna
    colunas_inconsistentes = []

    for col in df.columns:
        serie = df[col].dropna()
        if serie.empty:
            continue

        tipo = df[col].dtype
        valores = serie.astype(str).str.strip()

        # Define critérios de consistência com base no tipo declarado
        if pd.api.types.is_numeric_dtype(tipo):
            # Esperado: valores realmente numéricos
            valores_invalidos = valores[~valores.str.match(r'^-?\d+([.,]\d+)?$', na=False)]
            if len(valores_invalidos) / len(valores) > 0.1:
                colunas_inconsistentes.append(col)

        elif pd.api.types.is_datetime64_any_dtype(tipo):
            # Esperado: valores com padrão de data
            date_pattern = r'^\d{2}[/-]\d{2}[/-]\d{4}$|^\d{4}[/-]\d{2}[/-]\d{2}$'
            valores_invalidos = valores[~valores.str.match(date_pattern, na=False)]
            if len(valores_invalidos) / len(valores) > 0.1:
                colunas_inconsistentes.append(col)

        elif pd.api.types.is_object_dtype(tipo):
            # Esperado: conteúdo não parecendo número nem data (ex: "João", "SP", etc.)
            numeros_detectados = valores.str.match(r'^-?\d+([.,]\d+)?$', na=False)
            datas_detectadas = valores.str.match(r'^\d{2}[/-]\d{2}[/-]\d{4}$|^\d{4}[/-]\d{2}[/-]\d{2}$', na=False)
            if (numeros_detectados.mean() > 0.5) or (datas_detectadas.mean() > 0.5):
                colunas_inconsistentes.append(col)

    tipo_correto_pct = 1 - (len(colunas_inconsistentes) / len(df.columns))
    scores["Consistência"] = max(1, round(tipo_correto_pct * 5))
    diagnostico_colunas["Consistência"] = colunas_inconsistentes

    # 4. PRECISÃO - Detecta outliers em colunas numéricas
    outlier_cols = []
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        outlier_score = 0
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR))
            if outliers.mean() > 0.3:
                outlier_cols.append(col)
            outlier_score += (1 - outliers.mean())
        scores["Precisão"] = max(1, round((outlier_score / len(numeric_cols)) * 5))
        diagnostico_colunas["Precisão"] = outlier_cols
    else:
        scores["Precisão"] = 5

    # 5. INTEGRIDADE - Verifica regras semânticas e formatos
    integridade_score = 0
    total_checks = 0
    integridade_detalhes = []

    for col in df.columns:
        col_checks = 0
        col_passes = 0
        col_problemas = []

        serie = df[col]
        col_lower = col.lower()

        # COLUNAS NUMÉRICAS
        if pd.api.types.is_numeric_dtype(serie):
            # Check 1: Negativos em campos que deveriam ser positivos
            if any(k in col_lower for k in ['valor', 'preço', 'quantidade', 'saldo']):
                if serie.min() < 0:
                    col_problemas.append("valores negativos indevidos")
                else:
                    col_passes += 1
                col_checks += 1

            # Check 2: Variáveis binárias
            if 'binário' in col_lower or 'flag' in col_lower:
                valores = set(serie.dropna().unique())
                if not valores.issubset({0, 1}):
                    col_problemas.append("valores fora de 0/1 em campo binário")
                else:
                    col_passes += 1
                col_checks += 1

        # COLUNAS DE TEXTO
        elif pd.api.types.is_object_dtype(serie):
            sample = serie.dropna().astype(str).sample(min(100, len(serie.dropna())), random_state=42)

            # Check 3: Capitalização
            if sample.str.islower().any() or sample.str.isupper().any():
                col_problemas.append("capitalização inconsistente")
            else:
                col_passes += 1
            col_checks += 1

            # Check 4: Códigos e documentos
            if any(k in col_lower for k in ['cpf', 'cnpj', 'telefone', 'cep', 'id', 'cod', 'cd']):
                cleaned = sample.str.replace(r'\D', '', regex=True)

                expected_length = (
                    11 if 'cpf' in col_lower else
                    14 if 'cnpj' in col_lower else
                    8 if 'cep' in col_lower else
                    10 if 'telefone' in col_lower else
                    None
                )

                if any(k in col_lower for k in ['id', 'cod', 'cd']):
                    if cleaned.str.len().nunique() > 1:
                        col_problemas.append("tamanhos inconsistentes em códigos/IDs")
                    else:
                        col_passes += 1
                elif expected_length:
                    if not cleaned.str.len().eq(expected_length).all():
                        col_problemas.append(f"tamanho incorreto para {col}")
                    else:
                        col_passes += 1
                col_checks += 1

            # Check 5: Datas em formato texto
            if any(k in col_lower for k in ['data', 'dt', 'date', 'nascimento', 'inicio', 'fim']):
                date_pattern = r'^\d{2}[/-]\d{2}[/-]\d{4}$|^\d{4}[/-]\d{2}[/-]\d{2}$'
                valid_dates = sample.str.contains(date_pattern, na=False)
                if valid_dates.mean() < 0.9:
                    col_problemas.append("datas fora do padrão")
                else:
                    col_passes += 1
                col_checks += 1

        # COLUNAS DATETIME
        elif pd.api.types.is_datetime64_any_dtype(serie):
            col_passes += 1
            col_checks += 1

        # Soma pontuação
        if col_checks > 0:
            integridade_score += (col_passes / col_checks)
            total_checks += 1
            if col_problemas:
                integridade_detalhes.append(f"{col}: {', '.join(col_problemas)}")

    scores["Integridade"] = max(1, round((integridade_score / total_checks) * 5)) if total_checks else 5
    diagnostico_colunas["Integridade"] = integridade_detalhes

    return scores, diagnostico_colunas

# Upload de dados
uploaded_file = st.file_uploader("Carregue sua base de dados", type=["csv", "xlsx", "parquet", "json"])

# Leitura com tratamento para múltiplos formatos
if uploaded_file:
    try:
        file_name = uploaded_file.name.lower()
        file_extension = file_name.split('.')[-1]
        
        st.info(f"📄 Arquivo detectado: {file_name}")
        
        if file_extension == 'csv':
            encoding = detectar_encoding(uploaded_file)
            st.info(f"🔠 Encoding detectado: {encoding}")
            
            separador = detectar_separador(uploaded_file, encoding)
            st.info(f"🔧 Separador detectado: '{separador}'")
            
            df = pd.read_csv(uploaded_file, encoding=encoding, sep=separador)
            
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
            
        elif file_extension == 'parquet':
            # Usar BytesIO para leitura em memória
            from io import BytesIO
            df = pd.read_parquet(BytesIO(uploaded_file.getvalue()))
            
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
            
        else:
            raise ValueError("Formato de arquivo não suportado")
            
        st.success(f"✅ Dados carregados com {len(df)} linhas e {len(df.columns)} colunas")
        
    except Exception as e:
        st.error(f"❌ Erro na leitura: {str(e)}")
        st.error("""
        Dicas para correção:
        1. Verificar se o arquivo não está corrompido
        2. Para JSON: valide a estrutura em https://jsonlint.com
        3. Para Parquet: confirir a compatibilidade da versão
        """)
        st.stop()

    # Seção 1: Score de Qualidade
    st.header("🔍 Diagnóstico de Qualidade")
    scores, diagnostico_colunas = calcular_scores(df)
    score_final = round(np.mean(list(scores.values())), 1)
    
    def plot_radar_chart(scores):
        """Gera gráfico de radar com os scores de qualidade"""
        categories = list(scores.keys())
        values = list(scores.values())
        
        fig = px.line_polar(
            r=values + [values[0]],  # Fechar o círculo
            theta=categories + [categories[0]],
            line_close=True,
            range_r=[0, 5],
            title="Score de Qualidade da Base (1-5)"
        )
        fig.update_traces(fill='toself', line_color='#4591b8',fillcolor='rgba(69, 145, 184, 0.5)')
        fig.update_layout(
            polar=dict(
                bgcolor='white',
                radialaxis=dict(
                    visible=True,
                    range=[0, 5],
                    tickvals=[1, 2, 3, 4, 5],
                    color='black',
                    tickfont=dict(color='black')
                )
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black'),
            title_font=dict(color='black'),
            showlegend=False,
            height=400,
            # Movemos as margens para o layout principal
            margin=dict(t=30, b=30, l=30, r=30)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(f"Score Total: {score_final}/5")
        st.metric("Classificação", "⭐" * int(score_final))
        
        st.write("### Critérios:")
        
        # CSS para tooltip com ícone
        st.markdown("""
        <style>
        /* Tooltip com FontAwesome */
        .criteria-tooltip {
            position: relative;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 5px;
        }
        .criteria-tooltip .icon {
            color: #4591b8;
            font-size: 14px;
        }
        .criteria-tooltip .tooltip-text {
            visibility: hidden;
            width: 220px;
            background-color: #2c3e50;
            color: white;
            text-align: center;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.2s;
            font-size: 14px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .criteria-tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        /* Ajuste para não conflitar com seu progress bar */
        .stProgress {
            margin-top: -12px !important;
            margin-bottom: 8px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        descricoes_criterios = {
            "Completude": "Proporção de valores preenchidos (não nulos)",
            "Unicidade": "Presença de registros duplicados",
            "Consistência": "Valores batem com o tipo declarado",
            "Precisão": "Detecção de outliers numéricos",
            "Integridade": "Formato e semântica (CPF, datas, etc.)"
        }
        
        for criterio, valor in scores.items():
            st.markdown(f"""
            <div class="criteria-tooltip">
                {criterio}: {valor}/5
                <span class="icon"><i class="fas fa-info-circle"></i></span>
                <span class="tooltip-text">{descricoes_criterios[criterio]}</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(valor/5)

    with col2:
        st.write('<div style="margin-top:-20px;"></div>', unsafe_allow_html=True)  # Ajuste fino
        plot_radar_chart(scores)

    # Seção 2: Análise da IA
    if OPENAI_API_KEY:
        st.header("🧠 Análise e Recomendações da IA")
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        prompt = f"""

        🧱 FORMATO DA RESPOSTA (OBRIGATÓRIO):
        A IA **deve retornar as seções exatamente assim**:

        [PROBLEMA IDENTIFICADO]  
        [RECOMENDAÇÕES]  
        [MITIGAÇÃO]  

        🛑 Se a seção não tiver conteúdo, escreva:
        > "Nenhum problema identificado para esta seção"

        ---

        🚨 **REGRAS OBRIGATÓRIAS – LEIA CUIDADOSAMENTE ANTES DE RESPONDER:**

        - ❌ **NUNCA** analise critérios com score **maior que 3** 
        - ✅ **SÓ** analise critérios com **score 3, 2 ou 1**

        🛑 Se um critério tiver score **4 ou 5**, **NÃO escreva nada sobre ele**
        🛑 Mesmo que apareça no diagnóstico, ignore se o score dele for > 3
        🔍 Sempre cheque os scores antes de começar a análise!

        ---

        **Scores de qualidade (1-5) para esta base:**  
        {scores}

        **Critérios que PODEM ser analisados (score ≤3):**  
        { {k: v for k, v in scores.items() if v <= 3} }

        **Diagnóstico por Critério (colunas com problemas detectados):**  
        {diagnostico_colunas}

        ---

        **Critérios avaliados nesta análise:**

        1. **Completude** – Proporção de valores preenchidos (não nulos) nas colunas  
        2. **Unicidade** – Presença de registros duplicados (linhas repetidas)  
        3. **Consistência** – Verifica se os valores batem com o tipo de dado declarado (ex: número sendo string)  
        4. **Precisão** – Detecta valores extremos (outliers) em colunas numéricas  
        5. **Integridade** – Avalia se os dados seguem regras de formato e semântica (ex: CPFs, datas, flags binárias, capitalização(uso correto de letras maiúsculas e minúsculas em textos))

        ---

        **Contexto da base de dados:**
        - 📌 Número de colunas: {len(df.columns)}
        - 📌 Nomes das colunas: {list(df.columns)}
        - 📌 Tipos de dados principais: {dict(df.dtypes.value_counts())}

        Você é um analista de dados especializado em qualidade da informação. Responda com base nas instruções abaixo:

        ---

        [PROBLEMA IDENTIFICADO]

        1. A partir dos nomes das colunas, identifique o **domínio da base de dados** (ex: educação, saúde, financeiro, etc.). 

        2. **Para cada critério com scores menor ou igual a 3 (1, 2 ou 3)**:
            - 🧩 Explique **qual problema foi identificado**, com base no critério em questão
            - 🧠 Utilize **apenas o Diagnóstico por Critério** para indicar as colunas afetadas  
            - 📊 Descreva **como esse problema pode distorcer as análises no domínio identificado**
            - 💡 Dê **um exemplo prático** de como isso pode impactar uma transformação, cálculos e métricas

        ---

        [RECOMENDAÇÕES]

        Para cada problema detectado:

        1. 💡 Proponha uma **solução técnica específica** relacionada às colunas com problema  
        2. ⚙️ Indique o tipo de **transformação ou processamento** necessário (ex: padronização, conversão, tratamento)  
        3. 📈 Classifique a solução quanto ao **nível de complexidade**: baixo, médio ou alto

        Use blocos de código para soluções em pandas ou SQL.

        ---

        [MITIGAÇÃO]

        Para prevenir recorrência dos problemas:

        1. **🔍 Na Fase de Coleta (Prevenção Primária)**

            a) **Validações Críticas**:  
                - ✓ Regras de consistência para colunas relacionadas (ex: prc_aprovacao depende de alunos_previstos_LP)  
                - ✓ Campos obrigatórios em variáveis críticas (ex: prof_media_LP)  
                - ✓ Campos fechados (com regras) para códigos como sg_uf e nm_uf

            b) **Restrições de Formato**:  
                - ✓ Máscaras para campos como CPF, CNPJ, CEP  
                - ✓ Faixas de valor pré-definidas (ex: 0–100%)  
                - ✓ Tipagem estrita (ex: float, int, datetime)

        2. **⚙️ Durante o Processamento (Controle de Qualidade)**

            a) **🧹 Pipeline de Limpeza**:  
                - ✓ Padronização de encodings  
                - ✓ Tratamento de valores inconsistentes  
                - ✓ Registro de exceções para auditoria  
                - ✓ Padronização de formatos e nomes

        3. **✅ Pré-Exportação (Validação Final)**

            a) **✔️ Checklist obrigatório**:  
                - [ ] Verificar completude de campos críticos  
                - [ ] Validar consistência de tipos e formatos  
                - [ ] Confirmar aderência a regras de negócio

            b) **📋 Documentação de Regras**:
            ```markdown
            | Campo       | Restrição                       | Responsável |
            |-------------|----------------------------------|-------------|
            | data_nasc   | deve ser <= data_atual - 18 anos| RH          |
            | flag_bin    | deve conter apenas 0 ou 1       | Engenharia  |
            ```
        """

        with st.spinner("Gerando análise da IA..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": "Você é um especialista que transforma problemas de qualidade de dados em insights acionáveis específicos ao domínio analisado. Seja prático e direto."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3
                )
                content = response.choices[0].message.content
                
                # Função robusta para extrair seções
                def extract_section(content, section_tag):
                    start_tag = f"[{section_tag}]"
                    end_tags = [f"[{tag}]" for tag in ["PROBLEMA IDENTIFICADO", "RECOMENDAÇÕES", "MITIGAÇÃO"] if tag != section_tag]
                    
                    if start_tag in content:
                        section_content = content.split(start_tag)[1]
                        
                        # Encontra o próximo marcador de seção
                        end_positions = []
                        for tag in end_tags:
                            if tag in section_content:
                                end_positions.append(section_content.index(tag))
                        
                        if end_positions:
                            section_content = section_content[:min(end_positions)]
                        
                        return section_content.strip()
                    return f"Seção {section_tag} não encontrada na resposta da IA"

                # Extrai as seções
                problema = extract_section(content, "PROBLEMA IDENTIFICADO")
                recomendacoes = extract_section(content, "RECOMENDAÇÕES")
                mitigacao = extract_section(content, "MITIGAÇÃO")

                # Layout em três colunas
                st.markdown("""
                <style>
                    .analysis-card {
                        padding: 15px;
                        border-radius: 10px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        margin-bottom: 20px;
                        background: #dbebfa;  /* COR ALTERADA */
                        opacity: 0.7;        /* TRANSPARÊNCIA ADICIONADA */
                        height: 500px;
                        overflow-y: auto;
                        color: black !important;
                    }
                    .card-title {
                        font-size: 1.2em;
                        font-weight: bold;
                        margin-bottom: 15px;
                        color: #2c3e50;
                        border-bottom: 2px solid #0d0c0c;
                        padding-bottom: 8px;
                    }
                </style>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="analysis-card">
                        <div class="card-title">🛑 Problema Identificado</div>
                        {problema}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="analysis-card">
                        <div class="card-title">⚙️ Recomendações ETL</div>
                        {recomendacoes}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="analysis-card">
                        <div class="card-title">🛡️ Sugestões de Mitigação</div>
                        {mitigacao}
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Erro na geração da análise: {str(e)}")
                st.error("Por favor, tente novamente. Se o problema persistir, verifique sua conexão com a API da OpenAI.")

    # Seção 3: Data Profiling
    st.header("📊 Profiling Completo")

    # Gera e exibe o relatório
    with st.spinner("Gerando relatório de profiling..."):
        if len(df) > 100_000:
            st.warning("Usando amostra de 100k linhas para melhor performance")
            sample_df = df.sample(n=100000, random_state=42)
        else:
            sample_df = df
        
        profile = ProfileReport(
            sample_df,
            minimal=True,
            correlations={
                "pearson": {"calculate": True},
                "spearman": {"calculate": False},
                "kendall": {"calculate": False}
            }
        )
        
        # Exibe o relatório
        st.components.v1.html(profile.to_html(), height=800, scrolling=True)
        
        # Botão de download
        with st.expander("💾 Opções de Download", expanded=False):
            # CSS para botões lado a lado
            st.markdown("""
            <style>
            .download-row {
                display: flex;
                gap: 10px;
                margin-top: 10px;
            }
            .download-btn {
                flex: 1;
                background-color: white !important;
                border: 1px solid #008fcf !important;
                color: #008fcf !important;
                border-radius: 4px;
                padding: 8px;
                text-align: center;
                cursor: pointer;
                transition: all 0.2s;
            }
            .download-btn:hover {
                background-color: #f0f8ff !important;
            }
            /* Remove os botões padrão do Streamlit */
            .stDownloadButton { display: none !important; }
            </style>
            """, unsafe_allow_html=True)
            
            # Criar arquivo temporário para o relatório
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                profile.to_file(tmp.name)
                html_content = open(tmp.name, "rb").read()
            
            # Criar arquivo temporário para os dados
            csv_content = df.to_csv(index=False).encode('utf-8')
            
            # Layout dos botões - ÚNICA implementação necessária
            st.markdown("""
            <div class="download-row">
                <a href="data:text/html;base64,{b64_html}" download="profile_report.html" class="download-btn">
                    📥 Relatório (HTML)
                </a>
                <a href="data:text/csv;base64,{b64_csv}" download="dados_analisados.csv" class="download-btn">
                    📊 Dados (CSV)
                </a>
            </div>
            """.format(
                b64_html=base64.b64encode(html_content).decode(),
                b64_csv=base64.b64encode(csv_content).decode()
            ), unsafe_allow_html=True)

    # Seção 4: Recomendações Técnicas
    st.header("🛠️ Recomendações para Qualidade de Dados")

    # Criando abas para cada tipo de problema
    tab_comp, tab_uni, tab_cons, tab_int, tab_prec = st.tabs([
        "Completude", "Unicidade", "Consistência", "Integridade", "Precisão"
    ])

    with tab_comp:
        if scores["Completude"] <= 3:
            st.warning("**Problema Detectado**: Valores ausentes")
            st.markdown("""
            **O que significa?**  
            Colunas com valores nulos/em branco que podem prejudicar análises.
            
            **Como resolver?**  
            - Técnicas de imputação:
            ```python
            # Preencher com média (colunas numéricas)
            df['coluna'].fillna(df['coluna'].mean(), inplace=True)
            
            # Preencher com moda (colunas categóricas)
            df['coluna'].fillna(df['coluna'].mode()[0], inplace=True)
            ```
            - Coleta adicional dos dados faltantes
            - Exclusão se não forem críticos (`df.dropna()`)
            """)
        else:
            st.success("✅ Dados completos - Sem problemas significativos")

    with tab_uni:
        if scores["Unicidade"] <= 3:
            st.warning("**Problema Detectado**: Dados duplicados")
            st.markdown("""
            **O que significa?**  
            Registros idênticos que podem distorcer análises estatísticas.
            
            **Como resolver?**  
            ```python
            # Identificar duplicatas
            duplicates = df[df.duplicated(keep=False)]
            
            # Remover duplicatas (mantendo a primeira ocorrência)
            df.drop_duplicates(inplace=True)
            ```
            **Quando manter?**  
            Se forem registros válidos (ex: vendas do mesmo produto para o mesmo cliente em datas diferentes)
            """)
        else:
            st.success("✅ Dados únicos - Sem duplicatas problemáticas")

    with tab_cons:
        if scores["Consistência"] <= 3:
            st.warning("**Problema Detectado**: Inconsistência de tipos")
            st.markdown("""
            **O que significa?**  
            Dados armazenados em formatos incorretos (ex: números como texto).
            
            **Como corrigir?**  
            ```python
            # Converter tipos
            df['coluna'] = pd.to_numeric(df['coluna'], errors='coerce')  # Para números
            df['coluna'] = pd.to_datetime(df['coluna'], errors='coerce') # Para datas
            ```
            **Impacto:**  
            Cálculos estatísticos falham quando tipos estão inconsistentes.
            """)
        else:
            st.success("✅ Tipos consistentes - Dados bem formatados")

    with tab_int:
        if scores["Integridade"] <= 3:
            st.warning("**Problema Detectado**: Problemas de integridade")
            st.markdown("""
            **O que significa?**  
            Valores que violam regras de formato ou semântica:
            
            - **Códigos inválidos**: CPF/CNPJ/CEP com tamanho errado  
            - **Telefones**: Número incorreto de dígitos  
            - **IDs/Códigos**: Tamanhos inconsistentes  
            - **Binários**: Valores diferentes de 0/1  
            - **Números negativos**: Onde não são permitidos  
            - **Datas**: Futuras em campos históricos  

            **Como verificar manualmente:**  
            ```python
            # Verificar telefones (10 ou 11 dígitos)
            df['telefone'].str.replace(r'\D', '', regex=True).str.len().unique()
            
            # Verificar CEP (8 dígitos)
            df['cep'].str.replace(r'\D', '', regex=True).str.len().unique()
            
            # Verificar IDs (tamanho consistente)
            df['id'].astype(str).str.len().unique()
            
            # Verificar binários
            df['flag_binaria'].unique()  # Deve retornar apenas [0, 1]
            ```

            **Soluções recomendadas:**  
            1. Para **códigos mal formatados**:
            ```python
            # Padronizar CPF (123.456.789-09)
            df['cpf'] = df['cpf'].str.replace(r'(\d{3})(\d{3})(\d{3})(\d{2})', 
                                            r'\1.\2.\3-\4', regex=True)
            ```
            
            2. Para **valores binários inválidos**:
            ```python
            df['flag'] = df['flag'].apply(lambda x: 1 if x > 0 else 0)
            ```
            
            3. Para **IDs inconsistentes**:
            ```python
            # Preencher com zeros à esquerda
            df['id'] = df['id'].astype(str).str.zfill(10)  # 10 dígitos
            ```
            """)
        else:
            st.success("""
            ✅ Integridade validada - Todos os critérios:
            - Códigos (CPF/CNPJ/CEP/Telefone) com formatos corretos  
            - IDs/Códigos com tamanhos consistentes  
            - Variáveis binárias contendo apenas 0/1  
            - Sem valores negativos indevidos  
            """)

    with tab_prec:
        if scores["Precisão"] <= 3:
            st.warning("**Problema Detectado**: Outliers suspeitos")
            st.markdown("""
            **O que significa?**  
            Valores extremos que podem ser erros ou casos legítimos.
            
            **Como identificar?**  
            ```python
            # Método IQR
            Q1 = df['coluna'].quantile(0.25)
            Q3 = df['coluna'].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df['coluna'] < (Q1 - 1.5*IQR)) | (df['coluna'] > (Q3 + 1.5*IQR))]
            ```
            **Ações:**  
            - Verificar se são erros de medição/digitação  
            - Manter se forem casos válidos (ex: clientes premium)
            """)
        else:
            st.success("✅ Dados precisos - Sem outliers problemáticos")

    # Adicionando dicas gerais
    st.markdown("---")
    with st.expander("📚 Recursos Adicionais"):
        st.markdown("""
        - [Pandas: Tratamento de Dados Faltantes](https://pandas.pydata.org/docs/user_guide/missing_data.html)  
        - [SciKit-Learn: Imputação Avançada](https://scikit-learn.org/stable/modules/impute.html)  
        - [Artigo: Outliers Detection](https://medium.com/@gabrielpbreis/outliers-como-definir-detectar-e-tratar-parte-1-50bf8e5e229a)
        """)

    # Rodapé
    st.markdown("---")
    st.caption("Desenvolvido com Streamlit, ydata-profiling e OpenAI API • [Como contribuir?](https://github.com/daniell-santana/data-quality-profiling)")

# 📊 Data Profiling com IA - Análise Automática de Qualidade de Dados

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://data-quality-profiling-fmusyk6aoprwzclscnunjw.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
[![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=for-the-badge&logo=duckdb&logoColor=black)](https://duckdb.org/)

## 🎯 Objetivo
Ferramenta para análise automática de qualidade de dados que combina:
- **Data Profiling** tradicional
- **Avaliação de critérios de qualidade** (com scores 1-5)
- **Diagnóstico por IA** (LLM) com recomendações acionáveis
- **Visualização interativa** dos problemas

**Diferencial**:
- Explicações técnicas contextualizadas
- Detecção de anomalias
- Sugestões de correção especídficas e indicação de ações preventivas e corretivas

---

## 🧮 Como funciona o `calcular_scores`

Cada coluna do dataset é avaliada segundo os seguintes **critérios de qualidade**:

| Critério              | Descrição                                                                 |
|-----------------------|---------------------------------------------------------------------------------|
| **Completude**        | Proporção de valores não nulos                                                  |
| **Unicidade**         | Ausência de duplicatas em registros que deveriam ser únicos                     |
| **Consistência**      | Compatibilidade entre tipo declarado e conteúdo real                            |
| **Distribuição**      | Presença de outliers ou distorções estatísticas                                 |
| **Integridade**       | Detecção de inconsistências semâmtica, referncial e formato esperado da coluna  |

Cada critério recebe um **score de 1 a 5**, onde:
- **5** = qualidade excelente
- **3** = alerta, possível problema - ação preventiva
- **1** = grave, ação corretiva

Os scores são agregados em um diagnóstico final por coluna e armazenados em `df_scores`.

---

## 🤖 Diagnóstico com IA

Com base nos **scores calculados**, a IA (via API da OpenAI) gera um **diagnóstico automatizado**, seguindo dois princípios:

### 🔍 Intervenção Passiva (atual)
- A IA analisa colunas com `score ≤ 3` e sugere melhorias específicas.
- O foco está em **explicar os problemas** e **recomendar boas práticas** de forma contextualizada.
- Exemplo: se a completude for baixa, a IA pode sugerir imputação ou revisão da origem do dado.

### 🚀 Intervenção Ativa (futura melhoria)
- Implementar ações corretivas automáticas, como:
  - Preenchimento inteligente de valores faltantes
  - Correção de tipos de dados
  - Padronização de categorias
- A IA atuaria diretamente nos dados, com supervisão do usuário.

---

## 🛠️ Tecnologias Utilizadas

| Componente       | Descrição                             |
|------------------|---------------------------------------|
| `Streamlit`      | Interface web interativa              |
| `ydata-profiling`| Geração de relatórios automáticos     |
| `DuckDB`         | Processamento otimizado de dados      |
| `OpenAI API`     | Análise dos problemas                 |
| `Plotly`         | Visualizações dinâmicas               |
| `Pandas`         | Manipulação de dados                  |
| `Numpy`          | Realizar operações numéricas do scores|

---

## 🔄 Fluxo de Interação

```mermaid
flowchart TD
    A[UPLOAD DE DADOS] --> B{FORMATO DO ARQUIVO?}
    B -->|PARQUET/JSON| C[CARREGAMENTO DIRETO]
    B -->|CSV/XLSX| D[DETECTAR ENCONDING/SEPARADOR]
    
    D --> E[DATAFRAME PANDAS]
    C --> E
    
    E --> F[REGISTRAR NO DUCKDB]
    F --> G["CREATE TABLE df AS FROM pandas_df"]
    
    subgraph "🦆 DUCKDB"
        G --> H[METADADOS ANALYSE]
    end
    
    H --> I[TRANSFORMADO EM MAQUET]
    
    subgraph "🔍 CAMADA 1"
        I --> J[COMITÁRIOS\nvalores não nulos]
        I --> K[UNICIDADE\nregistros únicos]
        I --> L[CONSISTÊNCIA\ntipos de dados]
    end
    
    subgraph "🧠 CAMADA 2"
        I --> M[INTEGRIDADE\nvalores semelhantes e formatos]
        I --> N[PRECISÃO\noutliers numéricos]
    end
    
    %% Cálculo do Score
    J --> O[SCORE]
    K --> O
    L --> O
    M --> O
    N --> O
    
    %% Avaliação
    O --> P{Score ≤ 3?}
    P -->|Sim| Q[ACIONAR ANÁLISE DE IA]
    P -->|Não| R[RELATÓRIO PADRÃO]
    
    Q --> S[GPT-4: DIAGNÓSTICO ESPECÍFICO]
    S --> T[RECOMENDAÇÕES TÉCNICAS]
    S --> U[SUGESTÕES DE MITIGAÇÃO]
    
    R --> V[RELATÓRIO INTERATIVO]
    T --> V
    U --> V
    
    %% Saídas
    V --> W[VISUALIZAÇÕES]
    W --> X[RADAR DE SCORES]
    W --> Y[TABELA DE PROBLEMAS]
    W --> Z[AMOSTRA DE DADOS]
    
    V --> AA[EXPORTAÇÃO]
    AA --> BB[HTML: RELATÓRIO COMPLETO]
    AA --> CC[CSV: DADOS BRUTOS]
    
    AA --> DD[FEEDBACK LOOP]
    DD --> EE[ATUALIZAR REGRAS]
    EE --> H

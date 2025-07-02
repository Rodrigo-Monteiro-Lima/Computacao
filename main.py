import openai
import streamlit as st
import re
import os
import json
from typing import Optional, Dict

from dotenv import load_dotenv
load_dotenv(override=True)

from openai import OpenAI
st.set_page_config(page_title="IA Engenheira de Prompts", layout="centered")
st.markdown("""
<style>
/* Força a quebra de linha no st.code para ter o melhor dos dois mundos: botão de copiar e legibilidade */
div[data-testid="stCodeBlock"] pre {
    white-space: pre-wrap !important;
    overflow-wrap: break-word !important;
    word-wrap: break-word !important;
}
</style>
""", unsafe_allow_html=True)

client = None
try:
    print()
    client = OpenAI()
    if not client.api_key:
        st.error("Chave da API OpenAI não encontrada. Certifique-se de que a variável de ambiente OPENAI_API_KEY está configurada.")
        client = None
except Exception as e:
    st.error(f"Erro ao inicializar o cliente OpenAI: {e}")
    client = None

st.title("⚡️ IA Engenheira de Prompts Elétricos")

st.markdown("""
Descreva o **objetivo** do prompt que você quer criar. A IA irá gerar um prompt completo e otimizado para você usar em outras plataformas (ChatGPT, Claude, Gemini, etc.).
""")

def carregar_base_diretrizes():
    return {
        "otimizacao_fluxo_carga": {
            "descricao": "Otimização de Fluxo de Carga",
            "keywords_identificacao": ["fluxo de carga", "otimização de fluxo", "otimizar rede", "fluxo de potência", "despacho de geração", "reduzir perdas"],
            "prompt_guidelines": [
                "Definir a persona da IA como 'especialista em engenharia elétrica com foco em sistemas de potência'.",
                "Estruturar o prompt para solicitar informações essenciais como: Contexto do Sistema, Objetivo da Otimização, Dados Disponíveis e Métodos Preferenciais.",
                "Incluir uma instrução para a IA final 'pensar passo a passo' (Chain-of-Thought) antes de dar a resposta.",
                "Pedir que a resposta final seja estruturada, por exemplo, com um roteiro detalhado, lista de parâmetros e análise de premissas.",
                "Incorporar uma nota sobre a importância de usar princípios de engenharia e boas práticas da indústria."
            ]
        },
        "previsao_demanda_energia": {
            "descricao": "Previsão de Demanda de Energia",
            "keywords_identificacao": ["previsão de demanda", "prever carga", "demanda futura", "consumo futuro", "projeção de carga"],
            "prompt_guidelines": [
                "Definir a persona da IA como 'especialista em engenharia elétrica e análise de dados'.",
                "Estruturar o prompt para solicitar o escopo da previsão, período, dados históricos disponíveis, fatores a considerar e técnicas de interesse.",
                "Pedir que a IA final sugira uma metodologia passo a passo.",
                "Solicitar uma análise comparativa de diferentes técnicas de modelagem (prós e contras).",
                "Incluir uma instrução para a IA final discutir os principais desafios e como mitigá-los.",
                "Requerer que a IA final explique seu raciocínio e mencione as premissas importantes."
            ]
        }
    }

BASE_DIRETRIZES_EE = carregar_base_diretrizes()


def identificar_topico_por_keyword(descricao_usuario: str) -> Optional[str]:
    """
    Identifica o tópico de forma restritiva, baseado em keywords pré-definidas.
    """
    descricao_lower = descricao_usuario.lower()
    for topico_key, topico_data in BASE_DIRETRIZES_EE.items():
        for keyword in topico_data.get("keywords_identificacao", []):
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', descricao_lower):
                return topico_key
    return None

def gerar_prompt_com_ia(topico_identificado: str, descricao_usuario: str, modelo_ia: str = "gpt-4o") -> str:
    """
    Usa a IA para gerar um prompt otimizado com base no objetivo do usuário e em diretrizes.
    """
    if not client:
        return "ERRO: Cliente OpenAI não inicializado."

    info_topico = BASE_DIRETRIZES_EE[topico_identificado]
    diretrizes = info_topico["prompt_guidelines"]
    
    system_message = (
        "Você é uma IA especialista em Engenharia de Prompt, com profundo conhecimento em Engenharia Elétrica. "
        "Sua tarefa é gerar um prompt completo, detalhado, claro e otimizado para ser usado em outra IA (como GPT-4, Claude, etc.). "
        "O prompt que você criar deve ser robusto e capaz de extrair uma resposta técnica de alta qualidade da IA final."
    )
    
    diretrizes_str = "\n".join([f"- {d}" for d in diretrizes])
    user_message = (
        f"Preciso que você crie um prompt otimizado. Analise meu objetivo e as diretrizes para o tópico e gere o melhor prompt possível.\n\n"
        f"**Meu Objetivo:**\n\"{descricao_usuario}\"\n\n"
        f"**Tópico Identificado:**\n{info_topico['descricao']}\n\n"
        f"**Diretrizes para a Criação do Prompt:**\n{diretrizes_str}\n\n"
        "O prompt gerado deve ser auto-contido e pronto para ser copiado e colado. "
        "Ele deve preencher os detalhes mencionados no meu objetivo diretamente no corpo do prompt que você criar. "
        "Por exemplo, se meu objetivo menciona 'reduzir perdas em uma rede de 13.8kV', o prompt que você criar deve incluir explicitamente 'reduzir perdas' e 'rede de 13.8kV' nos locais apropriados."
    )

    try:
        response = client.chat.completions.create(
            model=modelo_ia,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.5,
            max_tokens=1024
        )
        prompt_gerado = response.choices[0].message.content
        return prompt_gerado

    except openai.APIError as e:
        return f"ERRO: A consulta à API da OpenAI para gerar o prompt falhou: {e}"
    except Exception as e:
        return f"ERRO: Um erro inesperado ocorreu ao gerar o prompt: {e}"


descricao_objetivo_usuario = st.text_area(
    "📝 **Descreva o objetivo do prompt que você quer criar:**",
    placeholder="Ex: Quero um prompt para analisar o fluxo de carga em minha rede de 13.8 kV com 55 barras, com o objetivo de reduzir as perdas elétricas.",
    height=150
)

if st.button("✨ Gerar Prompt Otimizado pela IA", use_container_width=True):
    if not client:
        st.error("O cliente OpenAI não pôde ser inicializado. Verifique sua API Key ou a conexão.")
    elif not descricao_objetivo_usuario.strip():
        st.warning("Por favor, descreva o objetivo do seu prompt.")
    else:
        with st.spinner("Analisando seu objetivo e elaborando o prompt... Por favor, aguarde."):
            topico_identificado = identificar_topico_por_keyword(descricao_objetivo_usuario)
            
            if not topico_identificado:
                topicos_conhecidos_descricoes = [f"'{data['descricao']}'" for data in BASE_DIRETRIZES_EE.values()]
                erro_msg = (f"Não foi possível identificar um tópico principal válido a partir da sua descrição. "
                            f"Por favor, inclua palavras-chave mais específicas relacionadas a um dos seguintes domínios: "
                            f"{' ou '.join(topicos_conhecidos_descricoes)}.")
                st.error(erro_msg)
            else:
                st.info(f"**Tópico identificado:** {BASE_DIRETRIZES_EE[topico_identificado]['descricao']}")
                
                prompt_otimizado = gerar_prompt_com_ia(topico_identificado, descricao_objetivo_usuario)
                
                if prompt_otimizado.startswith("ERRO:"):
                    st.error(prompt_otimizado)
                else:
                    st.success("Prompt otimizado gerado pela IA com sucesso!")
                    st.code(prompt_otimizado, language="markdown")

st.markdown("---")
st.caption("Copie o prompt gerado e use-o em sua plataforma de IA preferida para obter sua análise técnica.")
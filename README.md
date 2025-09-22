# 📊 Projeto de Feature Store & Predição de Churn  

Este projeto foi desenvolvido como parte dos meus estudos em Ciência de Dados, seguindo as aulas do canal [**Teo Me Why**](https://www.youtube.com/@TeoMeWhy) no YouTube.  

O objetivo principal é a construção de uma **Feature Store** utilizando **SQLite**, centralizando variáveis derivadas de transações, comportamento de clientes e interações com produtos. Dessa forma, os dados brutos são transformados em **features organizadas e reutilizáveis**, que podem ser consumidas diretamente em análises e modelos de Machine Learning.  

---

## 🚀 Funcionalidades Principais  

As queries SQL implementadas geram features que permitem análises e predições relacionadas ao **churn de clientes** e ao **engajamento com produtos**. Entre as principais variáveis:  

- **RFM (Recência, Frequência e Valor)**.  
- **Pontuação acumulada e resgatada**:
  - Ao longo da vida do cliente.  
  - Em janelas móveis de **7, 14 e 21 dias**.  
- **Interações com produtos específicos** (Chat, Lista de Presença, Resgates, Troca de Pontos, etc.), em valores absolutos e percentuais.  
- **Distribuição de pontos e transações por horário do dia** (manhã, tarde e noite).  
- **Engajamento em lives**:
  - Tempo médio, mínimo, máximo e total assistido.  
- **Flag de churn**, indicando se o cliente deixou de interagir após determinado período.  

---

## ⚙️ Como funciona  

- As **queries SQL** são parametrizadas por data de referência (`dtRef`).  
- Um script em Python (`execute.py`) é responsável por rodar as queries e atualizar a Feature Store.  
- As features são salvas no banco **SQLite** (`feature_store.db`), ficando disponíveis para análises, dashboards ou treinamento de modelos de Machine Learning.  

Exemplo de execução:  

```bash
# Atualizar a feature fs_points no período de 01/02/2024 até 01/06/2024
python execute.py -f fs_points -s 2024-02-01 -p 2024-06-01

Monitoramento do Modelo de Churn

Objetivo
Garantir que o modelo continue com bom desempenho ao longo do tempo e detectar drift de dados e performance.

Indicadores para monitorar:

1.Taxa de churn real mensal e prevista.

2.Distribuição de probabilidade prevista (média, p50, p75, p90).

3.Proporção de clientes por segmento (Críticos, Risco Médio, Estáveis).

4.Métricas de performance com rótulos disponíveis (Accuracy, Recall, Precision, F1 e AUC).

5.Drift em variáveis críticas: num_complaints, late_payments, contract_duration_months, app_usage_score, total_spent.

Regras de alerta:

1.Recall cair mais de 10% em relação ao baseline do teste.

2.AUC cair mais de 0.05 em relação ao baseline.

3.A proporção de clientes “Críticos” variar mais de 20% (relativo) por 2 semanas seguidas.

4.PSI maior que 0.2 em variáveis críticas (quando aplicável).

Frequência:

1.Monitoramento semanal de drift e distribuição.

2.Avaliação mensal com dados rotulados.

3.Retreino trimestral ou quando alertas dispararem por 2 ciclos consecutivos.

Rotina de atualização:

1.Coletar novos dados rotulados.

2.Reexecutar pipeline completo de treino.

3.Revalidar no conjunto de teste.

4.Versionar o modelo e atualizar models/churn_model.pkl e models/model_config.json.
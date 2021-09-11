from pomegranate import *

chuva = Node(DiscreteDistribution({
    "nenhuma": 0.9,
    "leve": 0.08,
    "pesada": 0.02
}), name="chuva")
#acidente é condicional a chuva
acidente = Node(ConditionalProbabilityTable([
    ["nenhuma", "sim", 0.6],
    ["nenhuma", "nao", 0.4],
    ["leve", "sim", 0.8],
    ["leve", "nao", 0.2],
    ["pesada", "sim", 0.9],
    ["pesada", "nao", 0.1]
], [chuva.distribution]), name="acidente")

lotacao2601 = Node(ConditionalProbabilityTable([
    ["nenhuma", "sim", "no horario", 0.8],
    ["nenhuma", "sim", "atrasada", 0.2],
    ["nenhuma", "nao", "no horario", 0.9],
    ["nenhuma", "nao", "atrasada", 0.1],
    ["leve", "sim", "no horario", 0.6],
    ["leve", "sim", "atrasada", 0.4],
    ["leve", "nao", "no horario", 0.7],
    ["leve", "nao", "atrasada", 0.3],
    ["pesada", "sim", "no horario", 0.4],
    ["pesada", "sim", "atrasada", 0.6],
    ["pesada", "nao", "no horario", 0.5],
    ["pesada", "nao", "atrasada", 0.5],
], [chuva.distribution, acidente.distribution]), name="lotacao2601")

#compromisso é condicional a lotação
compromisso = Node(ConditionalProbabilityTable([
    ["no horario", "presente", 0.9],
    ["no horario", "ausente", 0.1],
    ["atrasada", "presente", 0.6],
    ["atrasada", "ausente", 0.4]
], [lotacao2601.distribution]), name="compromisso")

model = BayesianNetwork()
model.add_states(chuva, acidente, lotacao2601, compromisso)

# Conexoes
model.add_edge(chuva, acidente)
model.add_edge(chuva, lotacao2601)
model.add_edge(acidente, lotacao2601)
model.add_edge(lotacao2601, compromisso)
model.bake()
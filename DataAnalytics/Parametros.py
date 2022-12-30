# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 09:04:54 2022

@author: AFICHER

Fichero con los parámetros del modelo. Variándolos se ajustan los resultados.
"""

#---------------- INTERWIDTH -----------------
# P valor máximo para considerar los features, si no lo hace ignora esta penalización
cs_InterWidth_minP = 0.000001

# la amplitud entre los centros de los features se divide entre la amplitud del intervalo. Posteriormente se eleva a este.
# un valor elevado hará que se desprecien más los intervalos menores
cs_InterWidth_power = 1.5

# ponderación de la amplitud del criterio de amplitud entre clusters
cs_InterWidth_beta = 1

# algunas cotas no tienen una amplitud definida. Para estos casos tomo este valor por defecto
cs_InterWidth_defaultInterval = 4

dict_CritScore = {
    "cs_InterWidth_power": cs_InterWidth_power,
    "cs_InterWidth_beta": cs_InterWidth_beta,
    "cs_InterWidth_defaultInterval": cs_InterWidth_defaultInterval,
    "cs_InterWidth_minP":cs_InterWidth_minP
}
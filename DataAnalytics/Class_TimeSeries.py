# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 08:33:53 2022

@author: AFICHER
"""

class TimeSeries:
    """
    Esta clase permite generar cálculos relacionados con tendencias lineales.
    P.e. cambios de tendencia, periodicidad, etc.
    """
    
    def __init__(self,X):
        self.X = X
        
        
        
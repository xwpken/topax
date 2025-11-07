'''

Heaviside projection (filter) for topology optimization

Last modified: 12/08/2024

'''

from dataclasses import dataclass

import jax
import jax.numpy as np
import matplotlib.pyplot as plt


class HeavisideGuest:
    
    '''
    Guest, James K., Jean H. Prévost, and Ted Belytschko. 
    "Achieving minimum length scale in topology optimization using nodal design variables and projection functions." 
    IJNME 61.2 (2004): 238-254.
    '''
    
    def __init__(self):
        self.citation = 'Guest2004'
        self.beta = 0
        self.num_update_beta = 0
        
    def compute(self, x):
        return (1. - np.exp(-self.beta*x) + x*np.exp(-self.beta))
        
    def set_params(self, beta):
        self.beta = beta
        self.num_update_beta += 1
        
        
class HeavisideWang:
    
    '''
    Guest, James K., Jean H. Prévost, and Ted Belytschko. 
    "Achieving minimum length scale in topology optimization using nodal design variables and projection functions." 
    IJNME 61.2 (2004): 238-254.
    '''
    
    def __init__(self):
        self.citation = 'Wang2010'
        self.num_update_beta = 0
        self.num_update_eta = 0
        
    def set_params(self, beta, eta):
        
        self.beta = beta
        self.num_update_beta += 1
        
        self.eta = eta
        self.num_update_eta += 1
     
    def compute(self,x):
        # tanh = (exp(x)-exp(-x))/(exp(x)+exp(-x))
        return ((
                np.tanh(self.beta*self.eta)+np.tanh(self.beta*(x-self.eta))
                )/(
                np.tanh(self.beta*self.eta)+np.tanh(self.beta*(1-self.eta))
                ))
    
    
@dataclass
class Projection:
        
    def visualize(self, param_dicts):
        params =self.extract_params(param_dicts)
        pojection_fns = self.trans_fns(*params)
        x = np.linspace(0.,1.,101)

        fig = plt.figure(figsize=(6,4))
        ax1 = fig.add_subplot()
        plt1 = ax1.plot(x,pojection_fns(x), color='#1f77b4', label='value')
        ax1.set_ylim([-0.1,1.1])
        ax1.set_xlabel('params')
        ax1.set_ylabel('value')
        
        ax2 = ax1.twinx()
        plt2 = ax2.plot(x,jax.vmap(jax.grad(pojection_fns))(x), 
                 color='#ff7f0e', label='derivative')
        ax2.set_ylabel('derivative')
        
        plts = plt1 + plt2
        labs = [p.get_label() for p in plts]
        ax1.legend(plts,labs)

        plt.title(f'{self.name}')
        plt.show()
        
        return fig

import numpy as np
import matplotlib.pyplot as plt

epsilonGeneral=0.1
gammaGeneral=1
alphaGeneral=0.1
class CliffWalking():
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto
        self.agentPos = [0, 0]
        # acciones
        self.arriba = 0
        self.abajo = 1
        self.derecha = 2
        self.izquierda = 3
        self.acciones = [self.arriba, self.abajo,
                         self.derecha, self.izquierda]
        
        # zonas
        self.startPos = [0, 3]
        self.goalPos = [11, 3]
    # end __init__
    
    def reset(self):
        self.agentPos = self.startPos
        return self.agentPos
    # end reset
    
    def actuar(self, accion):
        x, y = self.agentPos
        
        if(accion == self.arriba):
            y = y -1
            if(y<0):
                y = 0
        elif(accion == self.abajo):
            y = y +1
            if(y >= self.alto):
                y = self.alto -1
        elif(accion == self.derecha):
            x = x +1
            if(x >= self.ancho):
                x = self.ancho -1
        elif(accion == self.izquierda):
            x = x -1
            if(x<0):
                x = 0
        else:
            print('Accion desconocida')
            
        estado = [x, y]        
        reward = -1
        # x [1;10]
        # y = 3
        # cliff
        if(accion == self.abajo and y == 2 
           and 1 <= x <= 10) or (
            accion == self.derecha 
            and self.agentPos == self.startPos): #empieza precipicio   
            
            reward = -100
            estado = self.startPos
            
        self.agentPos = estado
            
        return self.agentPos, reward
    # end actuar
    
class AgenteQLearning():
    def __init__(self, entorno, alpha = alphaGeneral, epsilon = epsilonGeneral, gamma = gammaGeneral):#0.99):
        self.entorno = entorno
        self.nEstados = [entorno.ancho, entorno.alto]
        self.nAcciones = 4

        # policy params
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = np.zeros([self.nEstados[0], self.nEstados[1], self.nAcciones])
        
        # print(self.Q)
    # end __init__
    
    #policy Epsilon-Greedy
    def seleccionarAccion(self, estado):
        #exploracion
        if np.random.rand() <= self.epsilon: #aleatorio
            return np.random.randint(self.nAcciones)
        #explotacion
        else: # mejor valor Q
            return np.argmax(self.Q[estado[0], estado[1], :])
    # end seleccionarAccion
    
    # td control
    def QLearning(self, estado, estado_sig, accion, reward):
        td_target = reward + self.gamma * np.max(self.Q[estado_sig[0], estado_sig[1], :])
        td_error = td_target - self.Q[estado[0], estado[1], accion]
        self.Q[estado[0], estado[1], accion] += self.alpha * td_error
    
    def entrenar(self, episodios):
        recompensas = []

        for e in range(episodios):
            estado= self.entorno.reset()
            recompensa = 0
            fin = False

            while not fin:
                accion = self.seleccionarAccion(estado)
                estado_sig, reward = self.entorno.actuar(accion)
                recompensa += reward                
                
                fin = self.entorno.goalPos == estado

                if not fin:
                    #actualizar valor Q
                    self.QLearning(estado, estado_sig, accion, reward)
                estado = estado_sig

            #print('Fin episodio {}, reward: {}'.format(e, recompensa))
            recompensas.append(recompensa)

        return recompensas
    # end entrenar

class AgenteSarsa():
    def __init__(self, entorno, alpha = alphaGeneral, epsilon = epsilonGeneral, gamma = gammaGeneral):#0.99):
        self.entorno = entorno
        self.nEstados = [entorno.ancho, entorno.alto]
        self.nAcciones = 4

        # policy params
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = np.zeros([self.nEstados[0], self.nEstados[1], self.nAcciones])
        
        # print(self.Q)
    # end __init__
    
    #policy Epsilon-Greedy
    def seleccionarAccion(self, estado):
        #exploracion
        if np.random.rand() <= self.epsilon: #aleatorio
            return np.random.randint(self.nAcciones)
        #explotacion
        else: # mejor valor Q
            return np.argmax(self.Q[estado[0], estado[1], :])
    # end seleccionarAccion
    
    # td control        
    def Sarsa(self, estado,accion,reward, estado_sig,accion_sig):
        td_target = reward + self.gamma * (self.Q[estado_sig[0], estado_sig[1], accion_sig]) #por la posicion x,y es que tiene 2 estados siguientes.
        td_error = td_target - self.Q[estado[0], estado[1], accion]
        self.Q[estado[0], estado[1], accion] += self.alpha * td_error
    
    def entrenar(self, episodios):
        recompensas = []

        for e in range(episodios):
            
            estado= self.entorno.reset()
            accion = self.seleccionarAccion(estado)
            
            recompensa = 0
            fin = False

            while not fin:
                #accion = self.seleccionarAccion(estado)
                estado_sig, reward = self.entorno.actuar(accion)
                accion_sig = self.seleccionarAccion(estado_sig)
                
                recompensa += reward                
                
                fin = self.entorno.goalPos == estado

                if not fin:
                    #actualizar valor Q
                    self.Sarsa(estado,accion,reward,estado_sig,accion_sig)
                estado = estado_sig
                accion = accion_sig
            #print('Fin episodio {}, reward: {}'.format(e, recompensa))
            recompensas.append(recompensa)

        return recompensas
    # end entrenar  

    
cantidadAgentes = 50
    
entorno = CliffWalking(12, 4)
qlearning = np.zeros(500)
sarsa = np.zeros(500)


for a in range(cantidadAgentes):
    print('entrenando agente', a)
    
    agente = AgenteQLearning(entorno)
    qlearning += agente.entrenar(500) 
    
    agente = AgenteSarsa(entorno)
    sarsa += agente.entrenar(500) 


qlearning /= cantidadAgentes
sarsa /= cantidadAgentes


plt.plot(sarsa, label='Sarsa')
plt.plot(qlearning, label='QLearning')

plt.xlabel('Episodios')
plt.ylabel('Recompensa promedio')
plt.ylim([-300, -5])
plt.legend()
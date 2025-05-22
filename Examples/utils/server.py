import numpy as np 

from .client import Client


def server_update(all_betas, all_gammas):
    # Average to updated global parameters
    z_beta = np.mean(all_betas, axis=0)
    z_gamma = np.mean(all_gammas, axis=0)
    
    return z_beta, z_gamma


class Server:
    
    def __init__(self, data, event_col, duration_col, n_clients, n_epochs):
        
        self.event_col = event_col
        self.duration_col = duration_col
        self.n_clients = n_clients
        n_epochs = n_epochs
        
        # Specific to FedADMM
        self.u_betas = None 
        self.u_gammas = None 

        self.participants = self._init_clients(data)
    
    def _init_clients(self, data):
        
        participants = []
        
        data_split_idx = np.array_split(np.arange(data.shape[0]), self.n_clients)
        for idx in data_split_idx:  
            # Create a client object 
            client = Client(
                data.iloc[idx], 
                n_knots=5, 
                n_epochs=5, 
                event_col=self.event_col, 
                duration_col=self.duration_col
            )
            # Apply data pre-processing 
            client.preprocess_data()
            # Initialize model and parameters 
            client.init_model()

            participants.append(client)
            
        return participants 
    
    def fit(self):
        """
        """
        losses = []
        for _ in range(self.n_epochs):
            
            all_beta, all_gamma, last_local_losses = [], [], []
            for client in self.participants:
                client.fit_model(self.beta, self.gamma)
                
                all_beta.append(client.model.beta)
                all_gamma.append(client.model.gamma)
                last_local_losses.append(client.model.losses[-1])
                
            losses.append(np.mean(last_local_losses))
            # Aggregate model parameters 
            z_beta, z_gamma = server_update(all_beta, all_gamma)
            
        # Global parameter estimates 
        self.beta = z_beta
        self.gamma = z_gamma
    
    def fit_fedadmm(self):
        """
        """
        self.losses = []
        for _ in range(self.n_epochs):
            
            all_beta, all_gamma, last_local_losses = [], [], []
            for i, client in enumerate(self.participants):
                
                client.fit_fed_admm(z_beta, z_gamma) 

                all_beta.append(client.beta)
                all_gamma.append(client.gamma)
                last_local_losses.append(client.loss)
            
            self.losses.append(np.mean(last_local_losses))
            # Aggregate model parameters 
            z_beta, z_gamma = server_update(all_beta, all_gamma)
        
        # Global parameter estimates 
        self.beta = z_beta
        self.gamma = z_gamma
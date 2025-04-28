import numpy as np 

from .client import Client


def server_update(all_betas, all_gammas):
    # Average to updated global parameters
    z_beta = np.mean(all_betas, axis=0)
    z_gamma = np.mean(all_gammas, axis=0)
    
    return z_beta, z_gamma


def server_update_fedadmm(all_betas, all_gammas, u_betas, u_gammas):
    # Average to get new z (global parameters)
    z_beta = np.mean(all_betas, axis=0)
    z_gamma = np.mean(all_gammas, axis=0)

    # Update duals
    new_u_betas = [u + (b - z_beta) for b, u in zip(all_betas, u_betas)]
    new_u_gammas = [u + (g - z_gamma) for g, u in zip(all_gammas, u_gammas)]

    return z_beta, z_gamma, new_u_betas, new_u_gammas


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
                client.fit_model()
                
                all_beta.append(client.model.beta)
                all_gamma.append(client.model.gamma)
                last_local_losses.append(client.model.losses[-1])
                
            losses.append(np.mean(last_local_losses))
            
            z_beta, z_gamma = server_update(all_beta, all_gamma)
            
            for client in self.participants:
                # Distribute aggregated params 
                client.set_params({"beta": z_beta, "gamma": z_gamma})
                
        # Global parameter estimates 
        self.beta = z_beta
        self.gamma = z_gamma
    
    # TODO QUESTION: Clients/models supposed to re-use local param estimates 
    # when fitted, rather than re-init beta and gamma to agg params? 
    def fit_fedadmm(self):
        """
        """
        self.losses = []
        for _ in range(self.n_epochs):
            
            local_betas, local_gammas, last_local_losses = [], [], []
            for i, client in enumerate(self.participants):
                
                client.fit_fed_admm(z_beta, z_gamma, u_betas[i], u_gammas[i])

                local_betas.append(client.model.beta)
                local_gammas.append(client.model.gamma)
                last_local_losses.append(client.model.losses[-1])
            
            self.losses.append(np.mean(last_local_losses))
            
            z_beta, z_gamma, u_betas, u_gammas = server_update_fedadmm(
                local_betas, local_gammas, u_betas, u_gammas
            )
        
        # Global parameter estimates 
        self.beta = z_beta
        self.gamma = z_gamma
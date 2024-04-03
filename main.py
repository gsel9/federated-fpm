""" To recap, our end-goal is to make Python-based FL frameworks 
available to Stata users. 

A first use-case is to estimate a Stata model using a framework 
called Vantage6. To figure out how we can approach this, I want 
to prototype a solution containing the relevant syntax.

Here I sketch the classical approach to FL. I figured it makes 
a more realistic illustration of the order of events by using 
python objects (hope you are familiar with that). 

Questions are where in this process and how (syntax) we can integrate 
Stata estimation procedures (ie, call Stata ML optimizer) and make 
the coefficients available to post-estimation procedures.
"""
import numpy as np 


class Server:
    """ Representation of the centralized server. 

    In the classical setting, the role of the server is 
    (1) to initialize and distribute model coefficients to 
    the clients; (2) aggregate coefficient estimates recieved 
    from each client and re-distribute in an iterative process.
    """
    def __init__(self, p):
        # number of covaraites in each client's data 
        self.p = p 
        
        # placeholder for client's coefficients 
        self.client_coefs = []
                
    def init_coefs(self):
        " Initalialize coefficient estimates"
        self.coefs = np.random.random(self.p)

    def aggregate_client_coefs(self, client_coefs):
        """ Aggregate coefficient estimates by averaging 
        over the clients. 
        """ 
        self.coefs = np.mean(client_coefs, axis=1)
        
    def append_coefs(self, coefs):
        """ Recieve coefficients from a single client."""
        self.client_coefs.append(coefs)
    
    def distribute_coefs(self, clients):
        """ Distribute the server-side coefficients to each client. 
        """
        for client in clients:
            # update the coefficients at each client 
            client.recieve_coefs(self.coefs)
    
    
class Client:
    """ Representation of a client. 
    
    The client role is to iteratively (1) recieve a  
    coefficients estimate from the server; (2) update the 
    coefficients though optimization on local data; (3) 
    return the updated coefficients to the server. 
    
    A client can sometimes run multiple steps of optimization.
    """
    def __init__(
        self, coefs, optimizer=None, n_local_rounds=5
    ):
        self.coefs = coefs
        self.optimizer = optimizer
        self.n_local_rounds = n_local_rounds
        
    def update_coefs(self):
        # load local data 
        data = self.load_data()
        
        for _ in range(self.n_local_rounds):
            # run one optimization step on local data
            self.coefs = self.optimizer.run_step(data, self.coefs)
            
    def recieve_coefs(self, coefs):
        self.coefs = coefs
                
    def return_coefs(self, server):
        """ Pass the coefficients from the client to the server"""
        server.append_coefs(self.coefs)
            
    
def main():
    """Here we simulate the order of events in a FL experiment.""" 
    
    # pretend a server and 3 clients participating 
    server = Server()
    clients = [Client()] * 3
    
    # server initializes coefficients  
    server.init_coefs()
    
    # number of iterations at the server 
    n_server_rounds = ...
    for _ in range(n_server_rounds):
        
        # in each server round, the clients repeat the steps 
        for client in clients:        

            # update coefficients on client data 
            client.update_coefs()
            # return updated coefficients to server 
            client.return_coefs(server)
        
        # server aggregates coefficients over clients 
        server.aggregate_client_coefs()

        # server sends aggregated estimates back to clients  
        server.distribute_coefs(clients)
                
    
if __name__ == "__main__":
    main()
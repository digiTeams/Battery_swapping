import torch
import torch.nn as nn
import torch.nn.functional as F
from config import RL_params



class ValueNetwork(nn.Module):
    def __init__(self, state_dims: dict):
        '''
        State space at each time t:
        Agent: ('time_step', 'hour', 'cluster', 'nearby_stations', 'new_order',
                'cluster_demand', 'que_order', 'avg_waiting', 'full_batteries',
                'inventory_position')
        self.state_dims = {'steps': steps, 'hours': hours, 'stations': num_agents,
                           'clusters': num_clusters, 'observe_dim': 10,
                           'discrete': 3, 'continuous': 7}
        '''
        super(ValueNetwork, self).__init__()

        # One-hot for agent IDs
        self.num_agents = int(state_dims['stations'])
        self.num_steps = int(state_dims['steps'])
        self.num_hours = int(state_dims['hours'])
        self.num_clusters = int(state_dims['clusters'])
        self.num_continuous = int(state_dims['continuous'])

        # Embedding layers for discrete time features
        self.step_embed = nn.Embedding(
            num_embeddings=self.num_steps,
            embedding_dim=int(RL_params['step_embed_dims'])
        )
        ''' 
        self.hour_embed = nn.Embedding(
            num_embeddings=self.num_hours,
            embedding_dim=int(RL_params['hour_embed_dims'])
        )
        '''

        if RL_params['cluster_hot']:  #One-hot encoding
            id_dim = self.num_clusters
            self.station_embed = None

        else:  # Use cluster id embedding
            id_dim = int(RL_params['cluster_embed_dims'])
            self.station_embed = nn.Embedding(
                num_embeddings=self.num_clusters,
                embedding_dim=id_dim
            )

        # Input dimension after concat: step + hour embeddings, one-hot IDs, continuous
        input_dim = (
            int(RL_params['step_embed_dims']) + self.num_hours
            + id_dim + self.num_continuous
        )

        # Hidden layers
        layers = []
        prev_dim = input_dim
        for h in RL_params['hidden_dims']:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h
        self.hidden = nn.Sequential(*layers)

        # Output layer to scalar
        self.output_layer = nn.Linear(prev_dim, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: Tensor of shape (batch_size, 10) with columns:
        Agent: ('time_step', 'hour', 'cluster', 'nearby_stations', 'new_order',
                'cluster_demand', 'que_order', 'avg_waiting', 'full_batteries',
                'inventory_position')
        self.state_dims = {'steps': steps, 'hours': hours, 'stations': num_agents,
                           'clusters': num_clusters, 'observe_dim': 10,
                           'discrete': 3, 'continuous': 7}
        '''
        # Discrete features: must call ".long()" to get LongTensor
        timestep = x[:, 0].long()  # (batch,)
        hour     = x[:, 1].long()  # (batch,)
        ids      = x[:, 2].long()  # (batch,)

        # Embeddings
        step_embeddings = self.step_embed(timestep)  # (batch, step_embed_dims)
        hr_hots = F.one_hot(hour, num_classes=self.num_hours).float().to(x.device)
        if RL_params['use_station']:
            id_embed = self.station_embed(ids)
        else:
            id_embed = self.station_embed(ids)

        if RL_params['cluster_hot']:
            id_embed = F.one_hot(ids, num_classes=self.num_clusters).float().to(x.device)

        # Continuous features
        cont = x[:, 3:].float()  # (batch, num_cont)

        # Input
        features = torch.cat([step_embeddings, hr_hots, id_embed, cont], dim=1)

        # Hidden + output
        h = self.hidden(features)
        val = self.output_layer(h)

        # (batch,)
        return  val.squeeze(-1)

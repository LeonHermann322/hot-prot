import torch
factor = 8
linear_dim = 90112
linear_dim2 = int( linear_dim/ factor) 
linear_dim3 = int(linear_dim/ (factor*factor))
linear_dim4 = int(linear_dim/ (factor*factor*factor))

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(1, 512, (3,3), stride=2,padding=1 ),
                torch.nn.LayerNorm(512),
                torch.nn.GELU(),
                torch.nn.Conv2d(512, 256, (3,3), stride=2,padding=1 ),
                torch.nn.LayerNorm(256),
                torch.nn.GELU(),
                torch.nn.Conv2d(256, 128, (3,3), stride=2,padding=1 ),
                torch.nn.LayerNorm(128),
                torch.nn.GELU(),
                torch.nn.Conv2d(128, 32, (3,3), stride=2,padding=1 ),
                torch.nn.LayerNorm(64),
                torch.nn.GELU(),    
                torch.nn.Flatten()  
            )
     

        self.dense_encoder = torch.nn.Sequential(
            torch.nn.Linear(linear_dim, linear_dim2 ),
            torch.nn.LeakyReLU(),
        )
        self.mu_fc = torch.nn.Linear(linear_dim2, linear_dim3)
        self.sigma_fc = torch.nn.Linear(linear_dim2, linear_dim3)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to("cuda:0")
        self.N.scale = self.N.scale.to("cuda:0")
        self.kl = 0

    def forward(self, x):
        coded = self.encoder(x)
        latent = self.dense_encoder(coded)
        mu = self.mu_fc(latent)
        sigma = torch.exp(self.sigma_fc(latent))

        z = self.reparameterize(mu, sigma)
        self.kl = torch.mean(-0.2 * torch.sum(1 + sigma - mu ** 2 - sigma.exp(), dim = 1), dim = 0)
        return z

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
class Decoder(torch.nn.Module): 
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.dense_decoder = torch.nn.Sequential(
            torch.nn.Linear(linear_dim3, linear_dim2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(linear_dim2, linear_dim ),
            torch.nn.LeakyReLU(),
        )

        self.decoder = torch.nn.Sequential(
                torch.nn.Unflatten(1, (32,44,64)),
                torch.nn.ConvTranspose2d(32, 128, (3,3), stride=2, padding=1, output_padding=1),
                torch.nn.LayerNorm(128),
                torch.nn.GELU(),
                torch.nn.ConvTranspose2d(128, 256, (3,3), stride=2, padding=1, output_padding=1),
                torch.nn.LayerNorm(256),
                torch.nn.GELU(),
                torch.nn.ConvTranspose2d(256, 512, (3,3), stride=2, padding=1, output_padding=1),
                torch.nn.LayerNorm(512),
                torch.nn.GELU(),
                torch.nn.ConvTranspose2d(512, 768, (3,3), stride=2, padding=1, output_padding=1),
                torch.nn.LayerNorm(1024),
                torch.nn.GELU(), 
                torch.nn.ConvTranspose2d(768, 1, (1,1), stride=1, ),
                torch.nn.LayerNorm(1024),
                torch.nn.GELU(),    
            )
        
    def forward(self,x):
        decoded_laten = self.dense_decoder(x)
        decoded = self.decoder(decoded_laten)
        return decoded


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.encoder = Encoder().to("cuda:0")
        self.decoder = Decoder().to("cuda:1")

    
    def forward(self, x):
        x.to("cuda:0")
        encoded = self.encoder(x)
        bro = encoded.to("cuda:1")
        
        decoded = self.decoder(bro)
        return decoded
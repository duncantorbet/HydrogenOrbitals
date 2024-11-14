# Duncan Torbet
# 27/10/2024
# Computer Simulations - Project 2


####################################################################################
''' Solving the Shrodinger equation, in 3 dimensions, for a specified potential. '''
####################################################################################


## Library Import
import time
start = time.time() # This is just to measure the execution time of the file.

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy import sparse


# PyTorch Stuff
import torch # Importing Torch
from torch import lobpcg # This is used to compute eigenvalues and eigenvectors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # This will be used to send stuff to the GPU



## We begin by defining our constants:
pi = np.pi
hbar = 1/(2*pi) * 6.62607015e-34 # [m^2.kg.s^-1]
m = 9.10938371e-31 # [kg]
e = 1.60217663e-19 # [C]
e0 = 8.85418782e-12 # [C^2.kg^-1.m^-3.s^2]
J_to_eV = 6.241509e18 # [J/eV]
# hbar = 1
# m = 1
# e = 1
# e0 = 1
alpha = m/hbar**2

## Now we set up our meshgrid:
distance_scale = 25
distance_scale_real = distance_scale * 5.29e-11 # Bohr radius [m]
n = 100 # This will be the no. of steps within a given dimension. Increase this for precision, decrease this for better memory allocation.
x, y, z = (np.linspace(-distance_scale_real, distance_scale_real, n) for i in range(3))
X, Y, Z = np.meshgrid(x, y, z) # Our meshgrid of n steps for x, y & z.


## Now that we have our spacings, we can simlify the equation a little bit more:
dx = np.diff(X[0,:,0])[0] # This is the spacing of our discretization.
beta = alpha * dx**2


## Let's define our potential:
def potential(x, y, z):
    '''
    Here is the potential for a Hydrogen atom, this can be changed to whatever potential is desired.
    '''
    nonzero_distance = 1e-72 # This ensures that we never divide by 0
    r = np.sqrt(x**2 + y**2 + z**2 + nonzero_distance)
    return -1* beta * e**2 / (4 * pi * e0 * r)

V = potential(X, Y, Z)


## Now that we have everything set up, we can compute our Hamiltonian and put it into matrix form:
diag = -2*np.ones([n]) # Diagonal elements in the tridiag matrix.
offdiag = np.ones([n]) # Off-diagonal elements in the tridiag matrix.
all_diags = np.array([offdiag, diag, offdiag]) # Holding all the relevant diagonal values in an array.
diags_loc = np.array([-1,0,1]) # This is sppecifying the location of the diagonals, 0 being the main diagonal.

T = sparse.spdiags(all_diags, diags_loc, n, n) # Here we are creating a sparse Toeplitz matrix with the relevant diagonals (n**3 x n**3).
K = -0.5 * sparse.kronsum(sparse.kronsum(T, T), T) # This is the kronecker sum of all T matrices. Essentially this is the kinetic energy term.
U = sparse.diags(V.reshape(n**3),(0)) # Here we reshape our potential such that it has the correct dimensions.
H = K + U # This is our final Hamiltonian.
## We have our Hamiltonian, but we want to run this on the GPU, thus we are going to convert to PyTorch tensors and use cuda:

H.tocoo() # This converts our matrix to COOrdinate form. We need this in order to further convert to a torch tensor.
# print(H.nonzero())
inds, vals = np.array(H.nonzero()), np.array(H.data)
inds = torch.tensor(inds)
vals = torch.tensor(vals)
H = torch.sparse_coo_tensor(indices=inds, values=vals, size=H.shape).to(device)


## Now we have everything we need in order to determine the eigenvalues & eigenvectors using lobpcg:
# total_modes = int(input('How many (integer) orbital modes would you like to compute? \n'))
total_modes = 101
E_vals, E_vecs = lobpcg(H, k=total_modes, largest=False) # 'k' determines how many eigens you would like to return,
                                               # and 'largest=False' ensures we return the smallest eigens first
                 
E_vals = E_vals/beta # The eigenvalues were calculated as beta*E, so to find the physical energies, we need to divide by beta.
print(E_vals * J_to_eV)

def get_ev(amount): # This returns the n'th lowest eigenvector.
    return E_vecs.T[amount].reshape((n, n, n)).cpu().numpy()

## Now we can plot these eigenvectors, which correspond to the energies:
from skimage import measure
# mode = int(input('What orbital mode (integer) would you like? \n'))
for i in range(0,total_modes, int(total_modes/20)):
    # print(get_ev(mode)**2)
    vertices, faces, _, _, = measure.marching_cubes(get_ev(i)**2, level=1e-6) # This creates an iso-surface; a 2D surface in 3D,
                                                                                                # that corresponds to a chosen value.

    intensity = np.linalg.norm(vertices, axis=0) # This computes the distance from vertices to the origin, allowing for a colour gradient in our plot.

    import plotly.graph_objects as go
    plot_data = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], # This is setting the vertices for each dimension.
                        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], # This is setting the faces for each dimension.
                        colorscale='blackbody', opacity=0.5) # Setting the intensity, color & opacity. 

    fig = go.Figure(data=[plot_data]) # Plotting the data.

    ## This is to create the title of the n'th orbital.
    if int(i) == 0:
        suffix = 'th'
    elif int(i)%10 == 1:
        suffix = 'st'
    elif int(i)%10 == 2:
        suffix = 'nd'
    elif int(i)%10 == 3:
        suffix = 'rd'
    else:
        suffix = 'th'
    title = f'Hydrogen orbital for the {int(i)}{suffix} mode.'

    ## Removing all axes, backgrounds and grids etc.
    fig.update_layout(
        scene=dict(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                bgcolor='rgba(0,0,0,0)'
                ),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title=title
    )
    fig.show()






## Printing the execution time:

end = time.time()
execution_time = end - start
execution_time = np.round(execution_time, 0)
seconds = np.mod(execution_time, 60)
minutes = (execution_time - seconds)/60
print("Execution time: ", minutes, " minutes & ", seconds, " seconds." ,sep = "" )

#

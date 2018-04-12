# openMP_rectangles
This prog simulates a system of monodispersed long rectangles of size m x mk with no intersection allowed on a two dimensional lattice in grand canonical ensemble. It is a montecarlo simulation with heat bath dynamics where all the horizontal rectangles are evaporated and redeposited, then vertical rectangles and so on. In addition to the evaporation-deposition move, it alos flips a block of size mk x mk containing m vertical rectangles to horizontal ones and vice versa. This additional "flip" move expedites the equilibration.

The code requires two input files consisting of the list of probabilities for open and periodic boundary conditions. 

The random number generator is borrowed from GNU scientific library. It can generate different sequence of random numbers on differnt processors. The code has been parallelized using openMP. 

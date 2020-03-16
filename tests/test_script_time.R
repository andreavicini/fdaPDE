#########################################
#### TEST SCRIPT SPACE-TIME PROBLEMS ####
#########################################

library(fdaPDE)

####### 2D ########

#### square 2D , separable problem with mass penalization, spatial locations on mesh nodes ####

rm(list=ls())
graphics.off()

data(square2Ddata)

mesh=create.mesh.2D(nodes=nodes)
# x11()
plot(mesh)
# axis(1)
# axis(2)

FEMbasis=create.FEM.basis(mesh)

# Test function
set.seed(5847947)

a1=runif(1,min=-1.5,max=1.5)
a2=runif(1,min=-1.5,max=1.5)

z<-function(p)
{
  (a1*sin(2*pi*p[,1])*cos(2*pi*p[,2])+a2*sin(3*pi*p[,1]))*cos(p[,3])
  
}
NumTimePoints=11
TimePoints=seq(0,2,length.out = NumTimePoints)
# Exact solution (pointwise at nodes)
SpacePoints=mesh$nodes
SpaceTimePoints=cbind(rep(mesh$nodes[,1],NumTimePoints),rep(mesh$nodes[,2],NumTimePoints),rep(TimePoints,each=nrow(mesh$nodes)))

sol_exact=z(SpaceTimePoints)

ran=range(sol_exact) 

#image(FEM(sol_exact, FEMbasis))

# Set smoothing parameter

lambdaS = 1e-3
lambdaT = 1e-3
GCVFLAG=T

data = sol_exact + rnorm(length(sol_exact), mean=0, sd=0.05*(ran[2]-ran[1]))
observations=matrix(data,nrow(SpacePoints),NumTimePoints)

output_CPP<-smooth.FEM.time(time_mesh = TimePoints, observations=observations, FEMbasis=FEMbasis, lambdaS=lambdaS,
                            lambdaT=lambdaT, FLAG_MASS = TRUE, GCV=GCVFLAG)

image.FEM.time(output_CPP$fit.FEM.time,1)
#plot.FEM.time(output_CPP$fit.FEM.time,1)

#### simple mesh 2D (elliptic PDE, separable problem with identity penalization, covariates, 
#### spatial locations different from mesh nodes,temporal locations different from time mesh nodes) ####

rm(list=ls())
while (rgl.cur() > 0) { rgl.close() }

# Load the mesh
data(simpleMesh2Ddata)

mesh=create.mesh.2D(nodes=nodes, triangles = triangles, order=2)
plot(mesh)

# Create the FEM basis object
FEMbasis = create.FEM.basis(mesh)
time_mesh=seq(0,2,length.out = 5)
set.seed(5847947)

# Exact solution
f<-function(loc) {pi*sin(loc[,1])*loc[,3]}

data_exact=sin(pi*mesh$nodes[,1])

# Locations different from nodes
xobs=runif(min=-0.5,max=0.5,n=80)
yobs=runif(min=-0.5,max=0.5,n=80)
TimePoints=runif(min=0,max=2,n=5)
SpaceTimePoints=cbind(rep(xobs,5),rep(yobs,5),rep(TimePoints,each=80))

# Covariates - Locations different from nodes
cov1_nonod=sin(pi*SpaceTimePoints[,1])*SpaceTimePoints[,3]
cov2_nonod=rnorm(mean=0, sd=0.5,n=nrow(SpaceTimePoints))
W_nonod=cbind(cov1_nonod,cov2_nonod)

beta_exact=c(2,1)

# Exact data - locations different from nodes
data_exact = f(SpaceTimePoints)+W_nonod%*%beta_exact

# Perturbed data - locations different from nodes
data = data_exact + rnorm(n = length(data_exact), sd = 0.01)
observations=matrix(data,length(xobs),length(TimePoints))
# Set a vector of smoothing coefficients
lambdaS = 1e-3
lambdaT = 1e-3

GCVFLAG=FALSE

# Set PDE parameters
PDE_parameters_anys = list(K = matrix(c(0.01,0,0,1), nrow = 2), b = c(0,0), c = 0)

output_CPP = smooth.FEM.time(locations = cbind(xobs,yobs),time_locations = TimePoints, time_mesh = time_mesh,
                             observations = observations, covariates=W_nonod, FEMbasis = FEMbasis, 
                             lambdaS = lambdaS,lambdaT = lambdaT, PDE_parameters = PDE_parameters_anys,
                             GCV=GCVFLAG)
#image(output_CPP4$fit.FEM)
output_CPP$beta


#### charotid 2D (Parabolic problem (initial condition automatically estimated) + Boundary Conditions) ####

rm(list=ls())

data(charotid2Ddata)

plot(mesh)

FEMbasis = create.FEM.basis(mesh)
time_mesh=seq(0,4,length.out = 11)

# Set BC 
BC = NULL
BC$BC_indices = which(mesh$nodesmarkers == 1)
BC$BC_values = rep(0,length(BC$BC_indices))

lambdaS = 10^-6
lambdaT= 10^-6
set.seed(5839745)
DatiEsatti=rep(DatiEsatti,length(time_mesh))*rep(exp(time_mesh),each=length(DatiEsatti))
Data = DatiEsatti + rnorm(length(DatiEsatti), sd = 0.1)
observations=matrix(Data,nrow(SpacePoints),length(time_mesh))
GCVFLAG=T
GCVMETHODFLAG='Exact'

Sol = smooth.FEM.time(locations = SpacePoints, observations = observations,
                      time_mesh=time_mesh, FEMbasis = FEMbasis, FLAG_PARABOLIC = TRUE,
                      lambdaS = lambdaS, lambdaT = lambdaT, BC = BC, 
                      GCV=GCVFLAG,GCVmethod = GCVMETHODFLAG )
image(Sol$fit.FEM.time,1)  

plot(Sol$fit.FEM.time,1)


###### C mesh : parabolic problem (initial condition known) with GCV computation

rm(list=ls())
data(horseshoe2D)

mesh=create.mesh.2D(nodes = boundary_nodes, segments = boundary_segments)
mesh=refine.mesh.2D(mesh,30,0.03)
f<-function(x,y,t)
{
  K <- (y/0.1*as.double((abs(y)<=0.1 & x>-0.5))+as.double((abs(y)>0.1 | x<=-0.5)))^2
  res=numeric(length =length(x))
  for(i in 1:length(x))
  {
    if(x[i]>=0 && y[i]>0) 
      res[i]=cos(t[i])*(0.25*pi+x[i])+(y[i]-0.5)^2
    if(x[i]>=0 && y[i]<=0) 
      res[i]=cos(2*t[i])*(-0.25*pi-x[i])+(-y[i]-0.5)^2
    if(x[i]<0 && y[i]>0) 
      res[i]=cos(t[i])*(-atan(y[i]/x[i])*0.5)+(sqrt(x[i]^2+y[i]^2)-0.5)^2*K[i]
    if(x[i]<0 && y[i]<=0) 
      res[i]=cos(2*t[i])*(-atan(y[i]/x[i])*0.5)+(sqrt(x[i]^2+y[i]^2)-0.5)^2*K[i]
  }
  res
}


NumTimeInstants=5
TimePoints=seq(0,pi,length.out =NumTimeInstants)

K_func<-function(points)
{
  output = array(0, c(2, 2, nrow(points)))
  for (i in 1:nrow(points))
    output[,,i] = rbind(c(1,0),c(0,1))
  output
}


b_func<-function(points)
{
  output = array(0,c(2,nrow(points)))
  output
}

c_func<-function(points)
{
  rep(c(0), nrow(points))
}

u_func<-function(points)
{
  rep(c(0), nrow(points))
}
# Group all coefficients in one object
PDE_parameters = list(K = K_func, b = b_func, c = c_func, u = u_func)

FEMbasis = create.FEM.basis(mesh)
space_time_locations = cbind(rep(TimePoints,each=nrow(locations)),rep(locations[,1],NumTimeInstants),rep(locations[,2],NumTimeInstants))
DatiEsatti = f(space_time_locations[,2],space_time_locations[,3],space_time_locations[,1]) 

ICesatta = f(mesh$nodes[,1],mesh$nodes[,2],rep(0,nrow(mesh$nodes)))

Data = DatiEsatti+ rnorm(length(DatiEsatti), mean = 0, sd =  0.05*diff(range(DatiEsatti)))

lambdaS_Par = 10^seq(-7,3,length.out = 5) 
lambdaT_Par= 10^seq(-7,  3,length.out = 5)

solution =  smooth.FEM.time(locations = locations, time_mesh = TimePoints,
                            observations = matrix(data=Data[(dim(locations)[1]+1):(dim(locations)[1]*(length(TimePoints)))],nrow = nrow(locations),ncol = length(TimePoints)-1),
                            FEMbasis = FEMbasis,lambdaS = lambdaS_Par,lambdaT = lambdaT_Par,
                            FLAG_PARABOLIC = TRUE,IC = ICesatta, PDE_parameters = PDE_parameters,
                            GCVmethod = "Exact", GCV = T)
image.FEM.time(solution$fit.FEM.time,1,lambdaS = solution$bestlambda[1],lambdaT = solution$bestlambda[2])
plot.FEM.time(solution$fit.FEM.time,1,lambdaS =  solution$bestlambda[1],lambdaT = solution$bestlambda[2])


####### 2.5D #######

#### hub (covariates) ####

rm(list=ls())

data(hub25Ddata)

mesh <- create.mesh.2.5D(nodes = nodes,triangles = triangles)

FEMbasis <- create.FEM.basis(mesh)

# Locations at nodes
nodesLocations=mesh$nodes

# Exact data - Locations at nodes
nnodes = mesh$nnodes
a1 = rnorm(1,mean = 1, sd = 1)
a2 = rnorm(1,mean = 1, sd = 1)
a3 = rnorm(1,mean = 1, sd = 1)
TimeNodes = 0:4

locations = cbind(rep(TimeNodes,each=nnodes),rep(nodesLocations[,1],length(TimeNodes)),rep(nodesLocations[,2],length(TimeNodes)),rep(nodesLocations[,3],length(TimeNodes)))

func = function(x)
{
  (a1*sin(2*pi*x[,2])+a2*sin(2*pi*x[,3])+a3*sin(2*pi*x[,4])+1)*cos(x[,1])
}

func_evaluation = func(locations)
# Plot the exact solution
plot(FEM.time(coeff=array(func_evaluation,dim=c(length(func_evaluation),1,1)), FEMbasis = FEMbasis,time_mesh=TimeNodes,FLAG_PARABOLIC=T),TimeNodes)


lambdaS=10^seq(-9, -7, 0.5)
lambdaT=10^seq(-6, -4, 0.5)

lambdaS_par=10^seq(-4, -3, 0.25)
lambdaT_par=10^seq(1, 1.8, 0.2)

cov1=4*sin(2*pi*locations[,2])*cos(2*pi*locations[,3])
cov2=rnorm(nnodes*length(TimeNodes), mean=3, sd=0.1)*rep(exp(-TimeNodes/length(TimeNodes)),each=nnodes)
W=cbind(cov1,cov2)

# plot(FEM(coeff = cov1[1:nnodes], FEMbasis = FEMbasis))
# plot(FEM(coeff = cov2[1:nnodes], FEMbasis = FEMbasis))

# Fix betas
beta_exact=c(0.45,0.3)

ran=range(W%*%beta_exact + func_evaluation)
ran=range(func_evaluation)

# Plot exact solution
plot(FEM.time(coeff=array(W%*%beta_exact + func_evaluation, dim = c(nnodes*length(TimeNodes),1,1)),FEMbasis=FEMbasis,time_mesh = TimeNodes,FLAG_PARABOLIC = T),TimeNodes)

GCVFLAG=T
GCVMETHODFLAG='Stochastic'

ran = range(func_evaluation)
data = func_evaluation +rnorm(nnodes,mean=0,sd=0.05*(ran[2]-ran[1]))

ran = range(func_evaluation+ W%*%beta_exact)
datacov=func_evaluation+ W%*%beta_exact +rnorm(nnodes,mean=0,sd=0.05*(ran[2]-ran[1]))

data = matrix(data,mesh$nnodes,length(TimeNodes))
datacov = matrix(datacov,mesh$nnodes,length(TimeNodes))
#########################################SEPARABLE####################################################
solSep = smooth.FEM.time(locations=nodesLocations,observations = data,time_mesh = TimeNodes,
                         FEMbasis = FEMbasis, lambdaS = lambdaS, lambdaT = lambdaT, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC = F,nrealizations = 100,
                         GCVmethod = GCVMETHODFLAG)


solSepCov = smooth.FEM.time(locations=nodesLocations,observations = datacov,time_mesh = TimeNodes, covariates = W,
                            FEMbasis = FEMbasis, lambdaS = lambdaS, lambdaT = lambdaT, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC =F, nrealizations = 100,
                            GCVmethod = GCVMETHODFLAG)

##########################################PARABOLIC####################################################
solPar = smooth.FEM.time(locations=nodesLocations,observations = data,time_mesh = TimeNodes,
                         FEMbasis = FEMbasis, lambdaS = lambdaS_par, lambdaT = lambdaT_par, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC = T,nrealizations = 100,
                         GCVmethod = GCVMETHODFLAG)

solPar = smooth.FEM.time(locations=nodesLocations,observations = datacov[,2:length(TimeNodes)],time_mesh = TimeNodes, covariates = W[(1+mesh$nnodes):(length(TimeNodes)*mesh$nnodes),],
                         FEMbasis = FEMbasis, lambdaS = lambdaS_par, lambdaT = lambdaT_par, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC = T, nrealizations = 100,
                         IC=func_evaluation[1:mesh$nnodes],GCVmethod = GCVMETHODFLAG)




####### 2.5D #######

#### hub (covariates) areal####
rm(list=ls())

data(hub25DarealData)

nodesLocations=mesh$nodes

nnodes = mesh$nnodes
a1 = rnorm(1,mean = 1, sd = 1)
a2 = rnorm(1,mean = 1, sd = 1)
a3 = rnorm(1,mean = 1, sd = 1)
TimeNodes = 0:4
TimeNodesRMSE = seq(0,4,length.out = 15)

locations = cbind(rep(TimeNodesRMSE,each=nnodes),rep(nodesLocations[,1],length(TimeNodesRMSE)),rep(nodesLocations[,2],length(TimeNodesRMSE)),rep(nodesLocations[,3],length(TimeNodesRMSE)))

func = function(x)
{
  (a1*sin(2*pi*x[,2])+a2*sin(2*pi*x[,3])+a3*sin(2*pi*x[,4])+1)*cos(x[,1])
}

func_evaluation = func(locations)
FEMbasis=create.FEM.basis(mesh)

plot(FEM.time(coeff=array(func_evaluation,dim=c(length(func_evaluation),1,1)),time_mesh = TimeNodes,FEMbasis,FLAG_PARABOLIC = T),3)

sol_exact=func_evaluation

W_areal=cbind(rep(cov_areal,length(TimeNodes))*rep(exp(-TimeNodes/length(TimeNodes)),each=RDD_groups))

beta_exact=c(1)

lambdaS=10^seq(-9, -7, 0.5)
lambdaT=10^seq(-6, -4, 0.5)

lambdaS_par=10^seq(-5.2, -4.8, 0.1)
lambdaT_par=10^seq(1, 1.8, 0.2)

obs_areal = rep(obs_areal,length(TimeNodes))*rep(cos(TimeNodes),each=RDD_groups)

GCVFLAG=T
GCVMETHODFLAG='Stochastic'

ran = range(obs_areal)
data = obs_areal +rnorm(RDD_groups*length(TimeNodes),mean=0,sd=0.02*(ran[2]-ran[1]))

ran = range(obs_areal + W_areal%*%beta_exact)
datacov=obs_areal + W_areal%*%beta_exact + rnorm(RDD_groups*length(TimeNodes),mean=0,sd=0.02*(ran[2]-ran[1]))

data = matrix(data,RDD_groups,length(TimeNodes))
datacov = matrix(datacov,RDD_groups,length(TimeNodes))

###########################SEPARABLE###########################################
solSep = smooth.FEM.time(observations = data,time_mesh = TimeNodes,incidence_matrix = incidence_matrix,
                         FEMbasis = FEMbasis, lambdaS = lambdaS, lambdaT = lambdaT, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC = F,nrealizations = 100,
                         GCVmethod = GCVMETHODFLAG)

solSep = smooth.FEM.time(observations = datacov,time_mesh = TimeNodes, covariates = W_areal,incidence_matrix = incidence_matrix,
                         FEMbasis = FEMbasis, lambdaS = lambdaS, lambdaT = lambdaT, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC =F, nrealizations = 100,
                         GCVmethod = GCVMETHODFLAG)

##########################################PARABOLIC####################################################
solPar = smooth.FEM.time(observations = data[,2:length(TimeNodes)],time_mesh = TimeNodes, incidence_matrix = incidence_matrix,
                         FEMbasis = FEMbasis, lambdaS = lambdaS_par, lambdaT = lambdaT_par, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC = T,nrealizations = 100,
                         IC=func_evaluation[1:mesh$nnodes],GCVmethod = GCVMETHODFLAG)

solPar = smooth.FEM.time(observations = datacov[,2:length(TimeNodes)],time_mesh = TimeNodes, incidence_matrix = incidence_matrix,covariates = W_areal[(1+RDD_groups):(length(TimeNodes)*RDD_groups),],
                         FEMbasis = FEMbasis, lambdaS = lambdaS_par, lambdaT = lambdaT_par, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC = T, nrealizations = 100,
                         IC=func_evaluation[1:mesh$nnodes],GCVmethod = GCVMETHODFLAG)



####### 3D ########

#### sphere 3D (covariates + locations not at nodes + stochastic GCV) ####

rm(list=ls())

# Build mesh: Sphere
data(sphere3Ddata)
sphere3D<-create.mesh.3D(sphere3Ddata$nodes, sphere3Ddata$tetrahedrons)
# plot(sphere3D)
FEMbasis <- create.FEM.basis(sphere3D)
nodesLocations=sphere3D$nodes
nnodes = sphere3D$nnodes
TimeLocations = seq(0,1,length.out = 5)
Locations = cbind(rep(TimeLocations,each=nnodes),rep(nodesLocations[,1],length(TimeLocations)),rep(nodesLocations[,2],length(TimeLocations)),rep(nodesLocations[,3],length(TimeLocations)))

# Exact test function
a1 = rnorm(1,mean = 1, sd = 1)
a2 = rnorm(1,mean = 1, sd = 1)
a3 = rnorm(1,mean = 1, sd = 1)
a4 = rnorm(1,mean = 1, sd = 1)

func = function(x)
{
  a1*sin(2*pi*(x[,1]*x[,2]))+a2*cos(2*pi*x[,2])+a3*cos(2*pi*x[,3])+a4*sin(2*pi*x[,4])
}

func_evaluation = func(Locations)
ran=range(func_evaluation)
#
# plot(FEM(func_evaluation[1:nnodes],FEMbasis))
# Set smoothing parameter

# Generate locations
nloc = 1000
loc=matrix(data=runif(3*nloc, min=-1,max=1),nrow=nloc,ncol=3,byrow=T)

ind=NULL
for(row in 1:nloc){
  normvec = (loc[row,1]^2+loc[row,2]^2+loc[row,3]^2)
  if(normvec>0.975)   # check points outside the sphere and remove them
    ind = c(ind,row)
}

loc=loc[-ind,]
nloc=dim(loc)[1]
timeloc = seq(0,1,length.out=5)
loc = cbind(rep(timeloc,each=nloc),rep(loc[,1],length(timeloc)),rep(loc[,2],length(timeloc)),rep(loc[,3],length(timeloc)))


# Exact test function - locations different from nodes
func_evaluation2=func(loc)


cov1=(4*sin(2*pi*Locations[,2])+6*sin((2*pi*Locations[,3])^2))*(1-exp(-Locations[,1]))/3
cov2=cos(-2*pi*Locations[,4])+2*Locations[,1]*sin(2*pi*Locations[,2])/6

cov1_nonod=(4*sin(2*pi*loc[,2])+6*sin((2*pi*loc[,3])^2))*(1-exp(-loc[,1]))/3
cov2_nonod=cos(-2*pi*loc[,4])+2*loc[,1]*sin(2*pi*loc[,2])/6

W=cbind(cov1,cov2)
W2=cbind(cov1_nonod,cov2_nonod)



lambdaS=10^seq(-5.0, -4.0, 0.25)
lambdaT=10^seq(-1.5, -0.5, 0.25)

lambdaS2=10^seq(-5.5, -4.5, 0.25)
lambdaT2=10^seq(-1.5, -0.5, 0.25)

lambdaS_par=10^seq(-4.8, -4.4, 0.1)
lambdaT_par=10^seq(1.4, 1.8, 0.1)

lambdaS_par2=10^seq(-4.4, -4.0, 0.1)
lambdaT_par2=10^seq(1.4, 1.8, 0.1)

GCVFLAG=T
GCVMETHODFLAG='Stochastic'

ran = range(func_evaluation)
data = func_evaluation +rnorm(nrow(Locations),mean=0,sd=0.05*(ran[2]-ran[1]))

ran = range(func_evaluation2)
data_noloc = func_evaluation2 +rnorm(nrow(loc),mean=0,sd=0.05*(ran[2]-ran[1]))

ran = range(func_evaluation+ W%*%beta_exact)
datacov=func_evaluation+ W%*%beta_exact +rnorm(nrow(Locations),mean=0,sd=0.05*(ran[2]-ran[1]))

data = matrix(data,nnodes,length(TimeLocations))
data_noloc = matrix(data_noloc,nloc,length(timeloc))
datacov = matrix(datacov,nnodes,length(TimeLocations))
###########################SEPARABLE###########################################

solSep = smooth.FEM.time(locations=nodesLocations,observations = data,time_mesh = TimeLocations,
                         FEMbasis = FEMbasis, lambdaS = lambdaS, lambdaT = lambdaT, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC = F,nrealizations = 100,
                         GCVmethod = GCVMETHODFLAG)

solSepNoNodes = smooth.FEM.time(locations=loc[1:nloc,2:4],observations = data_noloc,time_mesh = timeloc,
                                FEMbasis = FEMbasis, lambdaS = lambdaS2, lambdaT = lambdaT2, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC = F,nrealizations = 100,
                                GCVmethod = GCVMETHODFLAG)

solSepCov = smooth.FEM.time(locations=nodesLocations,observations = datacov,time_mesh = TimeLocations, covariates = W,
                            FEMbasis = FEMbasis, lambdaS = lambdaS, lambdaT = lambdaT, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC =F, nrealizations = 100,
                            GCVmethod = GCVMETHODFLAG)

##########################################PARABOLIC####################################################
solPar = smooth.FEM.time(locations=nodesLocations,observations = data,time_mesh = TimeLocations,
                         FEMbasis = FEMbasis, lambdaS = lambdaS_par, lambdaT = lambdaT_par, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC = T,nrealizations = 100,
                         GCVmethod = GCVMETHODFLAG)

solParNoNodes = smooth.FEM.time(locations=loc[1:nloc,2:4],observations = data_noloc,time_mesh = timeloc,
                                FEMbasis = FEMbasis, lambdaS = lambdaS_par2, lambdaT = lambdaT_par2, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC = T,nrealizations = 100,
                                GCVmethod = GCVMETHODFLAG)

solParCov = fdaPDEtime::smooth.FEM.time(locations=nodesLocations,observations = datacov[,2:length(TimeLocations)],time_mesh = TimeLocations, covariates = W[(1+nnodes):(length(TimeLocations)*nnodes),],
                                        FEMbasis = FEMbasis, lambdaS = lambdaS_par, lambdaT = lambdaT_par, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC = T, nrealizations = 100,
                                        IC=func_evaluation[1:nnodes],GCVmethod = GCVMETHODFLAG)




####### 3D ########

#### sphere 3D areal (covariates + stochastic GCV) ####

rm(list=ls())

# Build mesh: Sphere
data(sphere3DarealData)
# plot(sphere3D)
FEMbasis <- create.FEM.basis(mesh)
nodesLocations=mesh$nodes
nnodes = mesh$nnodes
TimeLocations = seq(0,2,length.out = 15)
Locations = cbind(rep(TimeLocations,each=nnodes),rep(nodesLocations[,1],length(TimeLocations)),rep(nodesLocations[,2],length(TimeLocations)),rep(nodesLocations[,3],length(TimeLocations)))

# Exact test function
a1 = rnorm(1,mean = 1, sd = 1)
a2 = rnorm(1,mean = 1, sd = 1)
a3 = rnorm(1,mean = 1, sd = 1)
a4 = rnorm(1,mean = 1, sd = 1)

func_evaluation = numeric(nnodes)

func = function(x)
{
  a1*sin(2*pi*(x[,1]*x[,2]))+a2*cos(2*pi*x[,2])+a3*cos(2*pi*x[,3])*x[,1]^2+a4*sin(2*pi*x[,4])
}

func_evaluation = func(Locations)
ran=range(func_evaluation)
#
# plot(FEM(func_evaluation[1:nnodes+3*nnodes],FEMbasis))
# plot(FEM(vals[1:nnodes+3*nnodes],FEMbasis))

# Set smoothing parameter
# Generate areal data

obs_areal=NULL
sol_exact.FEM.time = FEM.time(coeff = array(func_evaluation,dim=c(length(func_evaluation),1,1)),time_mesh = TimeLocations,FEMbasis = FEMbasis,FLAG_PARABOLIC = T)
for(i in seq(0,2,length.out = 5))
  obs_areal=c(obs_areal,eval.FEM.time(FEM.time = sol_exact.FEM.time,incidence_matrix = incidence_matrix,lambdaS = 1,lambdaT = 1,locations = cbind(rep(i,RDD_groups))))

cov_areal=NULL
for(i in timeloc)
  cov_areal=c(cov_areal,eval.FEM.time(FEM.time = cov.FEM.time,incidence_matrix = incidence_matrix,lambdaS = 1,lambdaT = 1,locations = cbind(rep(i,RDD_groups))))

beta_exact=c(1.2)

W_areal=cbind(cov_areal)

lambdaS=10^-5
lambdaT=10^-5

lambdaS2=10^-5
lambdaT2=10^-5

lambdaS_par=10^-4.5
lambdaT_par=10^1

lambdaS_par2=10^-4.6
lambdaT_par2=1

GCVFLAG=F
GCVMETHODFLAG='Stochastic'


ran = range(obs_areal)
data = obs_areal +rnorm(length(obs_areal),mean=0,sd=0.05*(ran[2]-ran[1]))

ran = range(obs_areal + W_areal%*%beta_exact)
datacov=obs_areal + W_areal%*%beta_exact + rnorm(length(obs_areal),mean=0,sd=0.05*(ran[2]-ran[1]))

data = matrix(data,RDD_groups,length(timeloc))
datacov = matrix(datacov,RDD_groups,length(timeloc))
###########################SEPARABLE###########################################
solSep = smooth.FEM.time(observations = data,time_mesh = timeloc,incidence_matrix = incidence_matrix,
                         FEMbasis = FEMbasis, lambdaS = lambdaS, lambdaT = lambdaT, GCV = GCVFLAG, FLAG_MASS = T, FLAG_PARABOLIC = F,nrealizations = 100,
                         GCVmethod = GCVMETHODFLAG)

solSep = smooth.FEM.time(observations = datacov,time_mesh = timeloc,incidence_matrix=incidence_matrix ,covariates = W_areal,
                         FEMbasis = FEMbasis, lambdaS = lambdaS, lambdaT = lambdaT, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC =F, nrealizations = 100,
                         GCVmethod = GCVMETHODFLAG)

##########################################PARABOLIC####################################################
solPar = smooth.FEM.time(observations = data[,2:length(timeloc)],time_mesh = timeloc,incidence_matrix = incidence_matrix,
                         FEMbasis = FEMbasis, lambdaS = lambdaS_par, lambdaT = lambdaT_par, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC = T,nrealizations = 100,
                         IC=func_evaluation[1:nnodes],GCVmethod = GCVMETHODFLAG)

solPar = smooth.FEM.time(observations = datacov[,2:length(timeloc)],time_mesh = timeloc, covariates = W_areal[(1+nrow(incidence_matrix)):(length(timeloc)*nrow(incidence_matrix)),],incidence_matrix = incidence_matrix,
                         FEMbasis = FEMbasis, lambdaS = lambdaS_par, lambdaT = lambdaT_par, GCV = GCVFLAG, FLAG_MASS = F, FLAG_PARABOLIC = T, nrealizations = 100,
                         IC=func_evaluation[1:nnodes],GCVmethod = GCVMETHODFLAG)
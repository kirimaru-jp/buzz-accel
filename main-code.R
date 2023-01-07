# R code to reprodure the results in the paper 
# "Understanding narwhal diving behaviour using Hidden Markov Models with dependent state distributions and long range dependence" 
# [https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006425]
#
# We use Hidden Markov Models (HMMs) to label each dive of a narwhal based on 3 response variables of a dive
# "Maximum depth", "Dive dudation", and "Post-dive duration"
# We assume that these variables follow Independent/Dependent Log-normal and Gamma distributions, respectively
# Here is the code of fitting "Model 1" in the paper

#### 0. Process data ####

# genInp: create sequences interlaced by a sequence of postive interges and its opposites
# Input:   
# - x1: the first node
# - x2: the second node
# Output:  
# the interlaced sequence [0, x1, -x1, x1+1, -(x1+1), ..., x2, -x2]
genInp = function(x1,x2,st=1) {
	unique(c(0,unlist(lapply(seq(x1,x2,st),
							 function(x){c(x,-x)})))) }

# NumericHour: convert a POSIXCT datetime object to numeric
# Input:  
# - x: a POSIXCT datetime object
# Output: a numeric datetime
NumericHour = function(x) {
	library(lubridate); round((hour(x)+minute(x)/60+second(x)/3600),3)
}


options(digits = 12); library(data.table)
# Load R package to fit Model 1 in the paper, assumed the response variables following Log-normal distribution
library(plosM1.d3)

# Hyperparameter to define what a deep dive is, i.e. at least 350m, 400m, 500m, etc.
DEEP.DIVE = 350.0
# DEEP.DIVE = 400.0
# DEEP.DIVE = 500.0

deg_ns = 3 # degree of natural splines for covariates

# Read time-depth-recorder (TDR) dataset of ~ 83 days of a narwhal
dat0 = fread('data/PTT_3965_POSIX.csv', sep=',',fill=T,
			 header = T, colClasses=c(NA,'numeric')); head(dat0)

# Custom R package "mt" to find dive response variables 
# "Maximum depth", "Dive dudation", and "Post-dive duration"
# based on time-depth-recorder data
library(mt)

# FindDives(d, depth_begin, max_depth_min, bottom_percentage): find dives based on TDR data
# Input:
# - d: TDR time-series data
# - depth_begin: define when the dive begins
# - max_depth_min: maximum depth to decide whether it is a dive
# - bottom_percentage: the level to define the bottom phase of a dive
# Ouput: a list of dives found
L = mt::FindDives(dat0$Depth,0.0,20.0, 0.9)
detach('package:mt', unload=T, character.only=T)

# Convert a list of dives to a dataframe of dive
dat = data.table(data.frame(L)); dat = dat[Start!=0] ; 
dat$StartDepth = dat0$Depth[dat$Start]
dat$EndDepth   = dat0$Depth[dat$End]
dat$PostDiveDur = c(dat$Start[2:NROW(dat)]-dat$End[1:(NROW(dat)-1)],0); 
dat = dat[9:(NROW(dat)-1),] # skip first 8 dives -> reduce stress effect of tagging
# Convert unit of 3 response variables from seconds to minutes
dat[,c('BottDur','Duration','PostDiveDur')] = round(dat[,c('BottDur','Duration','PostDiveDur')]/60,3) 

# Make periodic spline of the covariate "onset of dive"
library(pbs)
M = matrix(pbs(NumericHour(dat0$Date[dat$Start]),
               df=3, Boundary.knots=c(0,24)),ncol=3)

head(dat); dim(dat); remove(L)

# Define the model: here is the Model 1 in the paer
model = 'plosM1.d3'; N = 3
# Initial value of the mean & standard deviation of 3 reponses variables for optimization procedure
## Max Depth ###
muMD0 = c(2.88, 5.01, 6.11)
sgMD0 = c(1.44, 0.65, 0.18)
## Duration ###
muDT0 = c(1.36, 1.79, 2.45)
sgDT0 = c(0.52, 0.40, 0.14)
## Post-Dive ###
muPD0 = c(0.68, 1.52, 1.75)
sgPD0 = c(0.335, 0.34, 1.25)
# Correlation between Max Depth and Duration of dives
Rho = rep(cor(log(dat$MaxDepth-19.5),log(dat$Duration)),N) # 
# Calculate covariates tau and d_t
dat$TotalDur   = dat$Duration + dat$PostDiveDur; head(dat)
dat$NoDDiveDur = noDdiveDur(dat, DEEP.DIVE)/60 # (min -> h): covariate tau
dat$DDive      = ddiveInrow(dat$MaxDepth, DEEP.DIVE) # covariate d_t

# Calculate natural splines of covariates 'tau' and 'd_t'
library(splines)
S = matrix(ns(dat$NoDDiveDur,df=3),ncol=3) # covariate tau
S_d = matrix(ns(dat$DDive,df=3),ncol=3)    # covariate d_t

library(digest)
digest(S, algo="md5", serialize=T)    # 7706d8b00e1a84121414e7958448cfdf
digest(S_d, algo="md5", serialize=T)  # f35accc2c425026799e46fbd45c914c1


#### 1. Find Maximum Likelihood ####
# Fit HMMs using multicore/parallel processing to find models with maximum likelihood

library(doMC); ncore = 16 # number of core for parallel computing
registerDoMC(ncore)

# Delete all old log files
# [https://stackoverflow.com/a/32998748]
unlink(paste0(model,'/check/*'),force=TRUE) 
unlink(paste0(model,'/*'),force=TRUE)

# Record time when fitting procedure start
s = proc.time(); st = Sys.time()

# Fitting models using various initial values
u = foreach(abg0=c(0,3,-3,5,-5),.combine=c) %:%
  foreach(abg1=genInp(1,5,2),.combine=c) %:%
  foreach(abg2=genInp(1,5,2),.combine=c) %dopar% {
  s1 = Sys.time()
  
  Alpha0 = Beta0 = Gamma0 = rep(mean(c(abg0,abg1,abg2)),N-1)
  Theta12 = Theta13 = rep(abg0,3)
  Theta21 = Theta23 = rep(abg1,3)
  Zeta31  = Zeta32  = rep(abg2,3)
  Delta12 = Delta13 = rep(mean(c(abg0,abg1,abg2)),3)
  Delta21 = Delta23 = rep(mean(c(abg0,abg1,abg2)),3)
  Delta31 = Delta32 = rep(mean(c(abg0,abg1,abg2)),3)
  
  params = pn2pw(muMD0,sgMD0,muDT0,sgDT0,muPD0,sgPD0,Rho,
            Alpha0, Beta0, Gamma0,
            Theta12, Theta13,
            Theta21, Theta23,
            Zeta31,  Zeta32,
            Delta12, Delta13,
            Delta21, Delta23,
            Delta31, Delta32)
  nom.fich = paste('chk',abg0,abg1,abg2,'txt', sep='.')
  cat(params, file=paste0(model,'/check/',nom.fich))

  tryCatch( {
    # Fitting using Newton method 'nlm'
    fHMM = mle(data=dat, params=params, N, M, S, S_d, print=0,optFunc=1)
    x = c(fHMM$mllk, nLogLike(params,dat,N,M,S,S_d),
        fHMM$muMD,  fHMM$sgMD,
        fHMM$muDT,  fHMM$sgDT,
        fHMM$muPD,  fHMM$sgPD,  fHMM$rho,
        fHMM$alpha0,fHMM$beta0, fHMM$gamma0,
        fHMM$theta12, fHMM$theta13, 
        fHMM$theta21, fHMM$theta23,
        fHMM$zeta31,  fHMM$zeta32,
        fHMM$delta12, fHMM$delta13, 
        fHMM$delta21, fHMM$delta23,
        fHMM$delta31, fHMM$delta32,
        difftime(Sys.time(),s1,units="mins"),fHMM$iteration)
    y = paste0(x,collapse = ",")
    cat("OK", file=paste0(model,'/check/',nom.fich))
    nom.fich = paste('res',abg0,abg1,abg2,'txt', sep='.')
    # Write the model output to log file
    cat(y, file=paste0(model,'/',nom.fich))
    y
  }, error=function(e) { tryCatch( {
    # Fitting using Newton method 'optim' if nlm failed
    fHMM = mle(data=dat, params=params, N, M,S,S_d, print=0,optFunc=2)
    x = c(fHMM$mllk, nLogLike(params,dat,N,M,S,S_d),
        fHMM$muMD,  fHMM$sgMD,
        fHMM$muDT,  fHMM$sgDT,
        fHMM$muPD,  fHMM$sgPD,  fHMM$rho,
        fHMM$alpha0,fHMM$beta0, fHMM$gamma0,
        fHMM$theta12, fHMM$theta13, 
        fHMM$theta21, fHMM$theta23,
        fHMM$zeta31, fHMM$zeta32,
        fHMM$delta12, fHMM$delta13, 
        fHMM$delta21, fHMM$delta23,
        fHMM$delta31, fHMM$delta32,
        difftime(Sys.time(),s1,units="mins"),
        as.numeric(fHMM$iteration)[1])
    y = paste0(x,collapse = ",")
    cat("OK", file=paste0(model,'/check/',nom.fich))
    nom.fich = paste('res',abg0,abg1,abg2,'txt', sep='.')
    # Write the model output to log file
    cat(y, file=paste0(model,'/',nom.fich))
    y
  }, error=function(e) {return('') }
  )	}	)
}

### Timing finished
message(cat('Running time (s):', sep = ''))
print(proc.time()-s)
# user     system    elapsed 
# 102678.539   4220.434   7293.982 

### Save global result ###
# Combine all model outputs and write to file
write(paste0("mllk,init_value,muMD_1,muMD_2,muMD_3,sgMD_1,sgMD_2,sgMD_3,",
			 "muDT_1,muDT_2,muDT_3,sgDT_1,sgDT_2,sgDT_3,",
			 "muPD_1,muPD_2,muPD_3,sgPD_1,sgPD_2,sgPD_3,rho1,rho2,rho3,",
			 'alpha0_0,alpha0_1,beta0_0,beta0_1,gamma0_0,gamma0_1,',
			 'theta12_1,theta12_2,theta12_3,',
			 'theta13_1,theta13_2,theta13_3,',
			 'theta21_1,theta21_2,theta21_3,',
			 'theta23_1,theta23_2,theta23_3,',
			 'zeta31_1,zeta31_2,zeta31_3,',
			 'zeta32_1,zeta32_2,zeta32_3,',
			 'delta12_1,delta12_2,delta12_3,',
			 'delta13_1,delta13_2,delta13_3,',
			 'delta21_1,delta21_2,delta21_3,',
			 'delta23_1,delta23_2,delta23_3,',
			 'delta31_1,delta31_2,delta31_3,',
			 'delta32_1,delta32_2,delta32_3,',
			 "runtime,iterations"),
	  paste0('output/',model,'.txt'),append=F)
outp=lapply(u, write, paste0('output/',model,'.txt'),append=T)


detach('package:plosM1.d3', unload=T, character.only=T)

### AIC values ####
# Show the AIC of the best of model based on AIC
res = fread(paste0('output/',model,'.txt')); setorder(res,mllk)
res = res[(muMD_1<muMD_2)&(muMD_2<muMD_3)]
ncol(res) - 4 # No. parameters: 63
round((res$mllk[1] + (ncol(res) - 4))*2, 2) # AIC: 170541.7

### 2.1 Decode & Viterbi #######
# Select the best model based on AIC then decode the states based on Viterbi algorithm
N = 3; ress = res[1,]

# Extract the paramteters of that best fitted model
muMD0  = as.numeric(ress[,c('muMD_1','muMD_2','muMD_3')])
sgMD0  = as.numeric(ress[,c('sgMD_1','sgMD_2','sgMD_3')])
muDT0  = as.numeric(ress[,c('muDT_1','muDT_2','muDT_3')])
sgDT0  = as.numeric(ress[,c('sgDT_1','sgDT_2','sgDT_3')])
muPD0  = as.numeric(ress[,c('muPD_1','muPD_2','muPD_3')])
sgPD0  = as.numeric(ress[,c('sgPD_1','sgPD_2','sgPD_3')])
Rho    = as.numeric(ress[,c('rho1','rho2','rho3')])

Alpha0 = as.numeric(ress[,c('alpha0_0','alpha0_1')])
Beta0  = as.numeric(ress[,c('beta0_0', 'beta0_1')])
Gamma0 = as.numeric(ress[,c('gamma0_0','gamma0_1')])

Delta12 = as.numeric(ress[,c('delta12_1','delta12_2','delta12_3')])
Delta13 = as.numeric(ress[,c('delta13_1','delta13_2','delta13_3')])
Delta21 = as.numeric(ress[,c('delta21_1','delta21_2','delta21_3')])
Delta23 = as.numeric(ress[,c('delta23_1','delta23_2','delta23_3')])
Delta31 = as.numeric(ress[,c('delta31_1','delta31_2','delta31_3')])
Delta32 = as.numeric(ress[,c('delta32_1','delta32_2','delta32_3')])

Theta12 = as.numeric(ress[,c('theta12_1','theta12_2','theta12_3')])
Theta13 = as.numeric(ress[,c('theta13_1','theta13_2','theta13_3')])
Theta21 = as.numeric(ress[,c('theta21_1','theta21_2','theta21_3')])
Theta23 = as.numeric(ress[,c('theta23_1','theta23_2','theta23_3')])

Zeta31 = as.numeric(ress[,c('zeta31_1','zeta31_2','zeta31_3')])
Zeta32 = as.numeric(ress[,c('zeta32_1','zeta32_2','zeta32_3')])

# Convert parameters to decode in Viterbi algorithm
params = pn2pw(muMD0,sgMD0,muDT0,sgDT0,muPD0,sgPD0,Rho,
      			   Alpha0, Beta0, Gamma0,
      			   Theta12, Theta13,
      			   Theta21, Theta23,
      			   Zeta31,  Zeta32,
      			   Delta12, Delta13,
      			   Delta21, Delta23,
      			   Delta31, Delta32)
# Decoding the states of each dive based on the fitted parameters
decoded_seq = viterbi(dat,N,pw2pn(params,N),M,S,S_d,DELTA = rep(1/N,N))
print(table(decoded_seq))
# decoded_seq
#   1    2    3 
# 3761 1932 2916 

# Weight according to the ratio of the decoded states
wght = as.vector(table(decoded_seq)/length(decoded_seq)); cat(wght*100)
dat$State=factor(decoded_seq)
dat[,c(3,4,6:8)] = NULL

# Save parameters to file, here with dependent Log-normal distribution
saveRDS(list( muMD=muMD0, sigmaMD=sgMD0,
               muDT=muDT0, sigmaDT=sgDT0,
               muPD=muPD0, sigmaPD=sgPD0, delta=wght ), 'data/dl.rds' )

# Load 4 "Model 1" fitted models, 
# where the responses variables follow Independent/Dependent Log-normal and Gamma distributions, respectively
DL = readRDS('data/dl.rds')
DG = readRDS('data/dg.rds')
IL = readRDS('data/il.rds')
IG = readRDS('data/ig.rds')
# Plot all the distribution together on the same histograms (Figure 5 on the paper)
source('drawing.R'); drawHisto_LN_GM(dat, DL,DG,IL,IG, 'fig5')

### 2.1.1 Table: Summary measures of Model 1 ####

# Compute the mean and standard deviation of each response variable 
# decoded in each state 1,2,3
SummModel = function(i) {
  df = dat[State == i]
  x= c(mean(df$MaxDepth), sd(df$MaxDepth),
    mean(df$Duration), sd(df$Duration),
    mean(df$PostDiveDur), sd(df$PostDiveDur), 
    Rho[i])
  round(x,2)
}
do.call(cbind,lapply(1:3, SummModel))
# [1,] 51.04 174.19 479.29
# [2,] 57.54 109.09  81.36
# [3,]  5.05   6.54  11.79
# [4,]  2.61   2.52   1.65
# [5,]  7.56   2.58   6.93
# [6,] 14.85   1.23   7.45
# [7,]  0.56   0.81   0.46

# Compute the percentage of each response variable 
# decoded in each state 1,2,3
SummDive = function(i) {
  df = dat[State == i]
  x = c(nrow(df)/nrow(dat)*100, 
        sum(df$Duration)/sum(dat$Duration)*100,
        sum(df$Duration)/sum(dat$TotalDur)*100,
        range(df$MaxDepth), range(df$Duration))
  round(x,2)
}
do.call(rbind,lapply(1:3, SummDive))
# [1,] 43.69 28.78 15.88  20.0 793.0 0.55 27.98
# [2,] 22.44 19.15 10.57  22.5 836.0 0.83 21.25
# [3,] 33.87 52.07 28.74 243.0 910.5 7.33 19.48


## Empirical distribution: test whether the state decoding decided solely by maximum depth
# take the data, and classify all dives below 350 meters as state 3, 
# all dives between 50 and 350 meters as state 2 and all dives below 50 meters as state 1
dat[,Emp_State:=1]
dat[MaxDepth > 50 & MaxDepth <= 350]$Emp_State = 2
dat[MaxDepth > 350]$Emp_State = 3
Emp_SummModel = function(i) {
  df = dat[Emp_State == i]
  x= c(mean(df$MaxDepth), sd(df$MaxDepth),
       mean(df$Duration), sd(df$Duration),
       mean(df$PostDiveDur), sd(df$PostDiveDur), 
       cor(df$MaxDepth,df$Duration),
       cor(df$MaxDepth,df$PostDiveDur),
       cor(df$PostDiveDur,df$Duration) )
  paste0('&',round(x,2))
}
noquote(do.call(cbind,lapply(1:3, Emp_SummModel)))


# ## Test effect of initial values
# Run, just one optimization, with model 1, correlated log-normal, 3 states, with initial values the optimized values, 
# only changing delta to the distribution given by the decoded state at time 0 (i.e., delta is a vector of 2 zeros and a 1, at the decoded state)
mod = mle(dat, params,N,M,S,S_d,print=2,optFunc=1,DELTA=c(1,0,0) ) # AIC: 170541.3093
dat$State2 = viterbi(dat,N,pw2pn(mod$argmax,N),M,S,S_d,DELTA=c(1,0,0))
dat$State = as.numeric(dat$State)
range(dat$State - dat$State2)
dat$Dive.Num = NULL
dat$TranState = c(diff(dat$State),-N)
df = dat[State == 3 & TranState == 0]; cor(df$MaxDepth,df$PostDiveDur)
df = dat[TranState == -2]; cor(df$MaxDepth,df$PostDiveDur)
df = dat[TranState == -1 & State == 2 ]; cor(df$MaxDepth,df$PostDiveDur)
df = dat[(TranState == -2) | ( TranState == -1 & State == 2 )]; cor(df$MaxDepth,df$PostDiveDur)

### 2.1.2 Viterbi & Histogram drawing ####
# Plot Viterbi decoding state and histogram of state distributions

source('drawing.R')
set(dat,j=c('Dive.Num'),value=seq_len(nrow(dat)))
# Plot Viterbi decoding state of dives 
drawFig(dat, N, wght,'Lognormal - Model 1','fig8')
# Plot histogram of Log-normal distribution
drawHisto(dat,muMD0,sgMD0,muDT0,sgDT0,muPD0,sgPD0,wght,'LN.model1.histo')

#### 2.2 95% Confident Interval ####
# Calculate the confident intervals of all fitted parameters

library(MASS)

params = pn2pw(muMD0,sgMD0,muDT0,sgDT0,muPD0,sgPD0,Rho,
               Alpha0, Beta0, Gamma0,
               Theta12, Theta13,
               Theta21, Theta23,
               Zeta31,  Zeta32,
               Delta12, Delta13,
               Delta21, Delta23,
               Delta31, Delta32)
params; nLogLike(params,dat,N,  M,S,S_d,DELTA = rep(1/N,N)) # 85207.8514517
# Fit model with obtained parameters to obtain Hessian matrix
fHMM = mle(data=dat, params=params, N, M,S,S_d, print=2,optFunc=1,DELTA = rep(1/N,N))
# Then compute the confidence intervals
Sigma = ginv(fHMM$H); var = diag(Sigma)
argm = fHMM$argmax
wlower = argm - 1.96*sqrt(var)
wupper = argm + 1.96*sqrt(var)
lower = pw2pn(wlower,N); upper = pw2pn(wupper,N)
L = lapply(lower,signif,digits = 3)
U = lapply(upper,signif,digits = 3)
ARG = lapply(pw2pn(argm,N),signif,digits = 3)


#### 2.3. QQ plots of 2-state, 3-state, 4-state ####
# Plot the QQ plots to see how the model fitted to real data of Model 1,
# in case there are 2, 3, and 4 states

library(cowplot)

# Convert QQ plot to data
qqDat = function(x) {
  as.data.table(setNames(qqnorm(x, plot.it=F), 
                         c("Theoretical", "Sample")))
}

# Plot the QQ-plots of Model 1 having 4 states
pr = readRDS('data/pseudoRes_M1-DL-4s.rds')
g1_4s = ggplot(qqDat(pr$MaxDepth) ) + geom_abline(size=2) + 
  geom_point(aes(x=Theoretical, y=Sample), size=3 ) +
  theme_classic(base_size=25) +
  theme(axis.text = element_text(color='black',size=25),
        plot.margin = unit(rep(0,4), "mm") ) +
  xlab(' ') + ylab( "4 states\n\n\n   " ) # + ggtitle(title)
g2_4s = ggplot(qqDat(pr$Duration) ) + geom_abline(size=2) + 
  geom_point(aes(x=Theoretical, y=Sample), size=3 ) +
  theme_classic(base_size=25) +
  theme(axis.text = element_text(color='black',size=25),
        plot.margin = unit(rep(0,4), "mm") ) +
  xlab('Theoretical') + ylab(" ") # + ggtitle(title)
g3_4s = ggplot(qqDat(pr$PostDiveDur) ) + geom_abline(size=2) + 
  geom_point(aes(x=Theoretical, y=Sample), size=3 ) +
  theme_classic(base_size=25) +
  theme(axis.text = element_text(color='black',size=25),
        plot.margin = unit(rep(0,4), "mm") ) +
  xlab(' ') + ylab(" ") # + ggtitle(title)

# Plot the QQ-plots of Model 1 having 3 states
pr = readRDS('data/pseudoRes_M1-DL-3s.rds')
g1_3s = ggplot(qqDat(pr$MaxDepth) ) + geom_abline(size=2) + 
  geom_point(aes(x=Theoretical, y=Sample), size=3 ) +
  theme_classic(base_size=25) +
  theme(axis.text = element_text(color='black',size=25),
        axis.title.y = element_text(margin = margin(t=0,r=-5,b=0,l=0) ),
        plot.margin = unit(rep(0,4), "mm") ) +
  xlab(' ') + ylab( bquote("3 states\n\n\nSample" ) ) # + ggtitle(title)
g2_3s = ggplot(qqDat(pr$Duration) ) + geom_abline(size=2) + 
  geom_point(aes(x=Theoretical, y=Sample), size=3 ) +
  theme_classic(base_size=25) +
  theme(axis.text = element_text(color='black',size=25),
        plot.margin = unit(rep(0,4), "mm") ) +
  xlab(' ') + ylab(" ") # + ggtitle(title)
g3_3s = ggplot(qqDat(pr$PostDiveDur) ) + geom_abline(size=2) + 
  geom_point(aes(x=Theoretical, y=Sample), size=3 ) +
  theme_classic(base_size=25) +
  theme(axis.text = element_text(color='black',size=25),
        plot.margin = unit(rep(0,4), "mm") ) +
  xlab(' ') + ylab(" ") # + ggtitle(title)

# Plot the QQ-plots of Model 1 having 2 states
pr = readRDS('data/pseudoRes_M1-DL-2s.rds')
g1_2s = ggplot(qqDat(pr$MaxDepth) ) + geom_abline(size=2) + 
  geom_point(aes(x=Theoretical, y=Sample), size=3 ) +
  theme_classic(base_size=25) +
  theme(axis.text = element_text(color='black',size=25),
        plot.margin = unit(rep(0,4), "mm"),
        plot.title = element_text(size=20)) +
  xlab(' ') + ylab( "2 states\n\n\n   " ) +
  ggtitle('Maximum Depth')
g2_2s = ggplot(qqDat(pr$Duration) ) + geom_abline(size=2) + 
  geom_point(aes(x=Theoretical, y=Sample), size=3 ) +
  theme_classic(base_size=25) +
  theme(axis.text = element_text(color='black',size=25),
        plot.margin = unit(rep(0,4), "mm"),
        plot.title = element_text(size=20) ) +
  xlab(' ') + ylab(" ") + ggtitle('Dive Duration')
g3_2s = ggplot(qqDat(pr$PostDiveDur) ) + geom_abline(size=2) + 
  geom_point(aes(x=Theoretical, y=Sample), size=3 ) +
  theme_classic(base_size=25) +
  theme(axis.text = element_text(color='black',size=25),
        plot.margin = unit(rep(0,4), "mm"),
        plot.title = element_text(size=20) ) +
  xlab(' ') + ylab(" ") + ggtitle('Post-dive Duration')


# Then join all the plots into one figure (Figure 9 in the paper)
pgrid = plot_grid(g1_2s,g2_2s,g3_2s, g1_3s,g2_3s,g3_3s, g1_4s,g2_4s,g3_4s,
                  ncol=3, nrow=3, align='vh',hjust=-1)
save_plot(paste0("figures/fig9.png"), pgrid,
          base_height = 12, base_width = 15, dpi = 300)


####  3. Plot dives and response variables based on decoded states ####

source('drawing.R')
library(Rcpp); library(RcppArmadillo); sourceCpp('functions.cpp')

# Draw dives in different colors associating to states during 12 hours (Figure 8)
s1 = (dat$Start[3945]-dat$Start[1]+1)-86400/4; s2 = (dat$Start[3945]-dat$Start[1]+1)+86400/4
ht0 = copy(dat0[s1:s2,]); ht0$Date = seq(s1,s2)
colnames(ht0) = c('DateTime','Depth')
ht0$ID = seq.int(nrow(ht0))/3600
range(ht0$Depth) ; head(dat)
ht0$State = as.factor(Depth2State(ht0$DateTime, ht0$Depth, dat))
head(ht0); table(ht0$State); 
drawDives.color(ht0)

# Plot response variables and covariate processes (Figure 3), 
# based on the decoded hidden states from a model fitted to a dependent log-normal distribution (Model 1).
HR = NumericHour(dat0$Date[dat$Start])
s1 = 3917-27; s2 = 3917+33
dat0 = copy(dat[s1:s2,]); dat0$NoDDiveDur = dat0$NoDDiveDur*60
dat0$Hour = HR[s1:s2]
drawState(dat0, N, wght, '')


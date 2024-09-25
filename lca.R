install.packages("reshape2")

library(MASS)
library(scatterplot3d)
library(poLCA)
library(dplyr)
library(reshape2)
library(ggplot2)
# Load the data and set subject id as the row names
cbcl_data <- read.csv("data/LCA/cbcl_t_no_mis_dummy.csv", row.names = 1)

# Specify the formula for latent class analysis
f <- with(cbcl_data, cbind(cbcl_scr_syn_anxdep_t, cbcl_scr_syn_withdep_t,
                            cbcl_scr_syn_somatic_t, cbcl_scr_syn_social_t,
                            cbcl_scr_syn_thought_t, cbcl_scr_syn_attention_t,
                            cbcl_scr_syn_rulebreak_t, cbcl_scr_syn_aggressive_t)~ 1)

# Class 4 results in a model with the lowest BIC (36349.93)

# Get the results of models with different class numbers into a dataframe
set.seed(01012)
lc2<-poLCA(f, cbcl_data, nclass=2, maxiter=50000, na.rm=FALSE,
              nrep=30, verbose=TRUE)
lc3<-poLCA(f, cbcl_data, nclass=3, maxiter=50000, na.rm=FALSE,
              nrep=30, verbose=TRUE)
lc4<-poLCA(f, cbcl_data, nclass=4, maxiter=50000, na.rm=FALSE,
              nrep=30, verbose=TRUE)
lc5<-poLCA(f, cbcl_data, nclass=5, maxiter=50000, na.rm=FALSE,
              nrep=30, verbose=TRUE)
lc6 <- poLCA(f, cbcl_data, nclass=6, maxiter=50000, na.rm=FALSE,
              nrep=30, verbose=TRUE)
lc7 <- poLCA(f, cbcl_data, nclass=7, maxiter=50000, na.rm=FALSE,
              nrep=30, verbose=TRUE)


# Save the models
save(lc2, lc3, lc4, lc5, lc6, lc7, file = "data/LCA/LCA_models_no_covariate.RData")

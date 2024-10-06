# install.packages("reshape2")

library(MASS)
library(scatterplot3d)
library(poLCA)
library(dplyr)
library(reshape2)
library(ggplot2)
# Load the data and set subject id as the row names
cbcl_data <- read.csv("data/LCA/final_sample_cbcl.csv", row.names = 1)

# Specify the formula for latent class analysis
f <- with(cbcl_data, cbind(cbcl_scr_syn_anxdep_t, cbcl_scr_syn_withdep_t,
                            cbcl_scr_syn_somatic_t, cbcl_scr_syn_social_t,
                            cbcl_scr_syn_thought_t, cbcl_scr_syn_attention_t,
                            cbcl_scr_syn_rulebreak_t, cbcl_scr_syn_aggressive_t)~ 1)


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
save(lc2, lc3, lc4, lc5, lc6, lc7, file = "data/LCA/LCA_models_final_sample.RData")


# Save the predicted class and posterior probabilities to the cbcl_data dataframe
# Also compute the individual level entropy

compute_individual_entropy <- function(posterior_probs) {
  # Ensure posterior_probs does not contain any zeroes before applying log
  posterior_probs <- ifelse(posterior_probs == 0, 1e-10, posterior_probs)
  entropy <- -sum(posterior_probs * log(posterior_probs))
  return(entropy)
}

cbcl_data$predicted_class <- lc4$predclass

posterior_matrix <- lc4$posterior

# Compute entropy for each individual
entropies <- apply(posterior_matrix, 1, compute_individual_entropy)

# Add entropies to cbcl_data
cbcl_data$entropy <- entropies

# Add the posterior probabilities as new columns in cbcl_data
posterior_columns <- paste("ClassProb", 1:ncol(lc4$posterior), sep="_")
cbcl_data[posterior_columns] <- lc4$posterior

# Save the data with the predicted class and posterior probabilities
write.csv(cbcl_data, "data/LCA/cbcl_final_class_member.csv", row.names = TRUE)
###

###
# Visualise conditional probability of each variable for each class
load("data/LCA/LCA_models_final_sample.RData")

lcmodel <- reshape2::melt(lc4$probs, level=2)
write.csv(lcmodel, "data/LCA/lcmodel_prob_class_final_sample.csv", row.names = TRUE)


lc4

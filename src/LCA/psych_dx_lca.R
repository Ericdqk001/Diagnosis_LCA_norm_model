install.packages("reshape2")

library(MASS)
library(scatterplot3d)
library(poLCA)
library(dplyr)
library(reshape2)
library(ggplot2)

psych_dx_data <- read.csv("data/LCA/psych_dx_dummy.csv", row.names = 1)

# Specify the formula for latent class analysis
f <- with(psych_dx_data, cbind(Has_ADHD,Has_Depression,Has_Bipolar,Has_Anxiety,Has_OCD,Has_ASD,Has_DBD)~ 1)

# Get the results of models with different class numbers into a dataframe
set.seed(01012)
lc2<-poLCA(f, psych_dx_data, nclass=2, maxiter=50000, na.rm=FALSE,
              nrep=30, verbose=TRUE)
lc3<-poLCA(f, psych_dx_data, nclass=3, maxiter=50000, na.rm=FALSE,
              nrep=30, verbose=TRUE)
lc4<-poLCA(f, psych_dx_data, nclass=4, maxiter=50000, na.rm=FALSE,
              nrep=30, verbose=TRUE)
lc5<-poLCA(f, psych_dx_data, nclass=5, maxiter=50000, na.rm=FALSE,
              nrep=30, verbose=TRUE)
lc6 <- poLCA(f, psych_dx_data, nclass=6, maxiter=50000, na.rm=FALSE,
              nrep=30, verbose=TRUE)

save(lc2, lc3, lc4, lc5, lc6, file = "data/LCA/LCA_models_with_control.RData")

# Entropy measure function
lca_re <- function(x) {
  nom <- (sum(-x$posterior*log(x$posterior)))
  denom <- (nrow(x$posterior)*log(ncol(x$posterior)))
  re <- 1 - (nom/denom)
  if (is.nan(re) == TRUE) re <- NA
  return(re)
}

# Initialize containers for the model statistics
out_lca <- list(lc2, lc3, lc4, lc5, lc6) # list of loaded models
npar <- ll <- bic <- abic <- caic <- awe <- re <- aic <- c() # containers for statistics

# Loop through each model and extract the relevant information
for (k in 1:length(out_lca)) {
  fit <- out_lca[[k]]
  npar[k] <- fit$npar
  ll[k]   <- fit$llik
  bic[k]  <- fit$bic
  aic[k]  <- fit$aic
  abic[k] <- -2 * (fit$llik) + fit$npar * log((fit$Nobs + 2) / 24)
  caic[k] <- -2 * (fit$llik) + fit$npar * (log(fit$Nobs) + 1)
  awe[k]  <- -2 * (fit$llik) + 2 * (fit$npar) * (log(fit$Nobs) + 1.5)
  re[k]   <- round(lca_re(fit), 3)
}

# Create a vector of class names
class <- paste0("Class-", 2:6)

# Store the information in a data frame
poLCA.tab <- data.frame("Class" = class, "Npar" = npar, "LL" = ll,
                        "AIC" = aic, "BIC" = bic, "aBIC" = abic, 
                        "CAIC" = caic, "AWE" = awe, "RE" = re)

# Print the data frame
print(poLCA.tab)

# Save the data frame as a CSV file
write.csv(poLCA.tab, file = "data/LCA/LCA_model_fit_statistics_with_control.csv", row.names = FALSE)

lcmodel <- reshape2::melt(lc4$probs, level=2)
write.csv(lcmodel, "data/LCA/lcmodel_prob_class.csv", row.names = TRUE)
install.packages(c("agridat", "ggplot2", "ghibli", "ggdist"))
library(ggplot2)   # Plotting
library(ggdist)    # For raincloud plot elements (dots, halfeye)
library(dplyr)     # Data manipulation
library(readr)  

data <- read.csv("src/discover/results/output_data_low_entropy/t1w_cortical_thickness_rois_output_data_with_dev.csv")


head(data)

data <- data %>%
  mutate(cohort = case_when(
    cohort1 == 1 ~ "low_symp_test_subsp",
    cohort2 == 1 ~ "inter_test_subs",
    cohort3 == 1 ~ "exter_test_subs",
    cohort4 == 1 ~ "high_test_subs"
  ))
install.packages("ggplot2")
install.packages("ggridges")
install.packages("tidyr")
install.packages("dplyr")

library(ggplot2)
library(ggridges)
library(tidyr)
library(dplyr)
# Load the csv results data

data <- read.csv("src/discover/results/t1w_cortical_thickness_rois_output_data.csv")

head(data)

data_long = pivot_longer(data, cols = low_symp_test_subs:high_test_subs, names_to = "Clinical_Cohorts", values_to = "value")

data_long = data_long[data_long$value == 1, ]  # Keep only rows where Value is 1


theme_set(theme_ridges())

# Calculate mean and median for each cohort
summary_stats <- data_long %>%
  group_by(Clinical_Cohorts) %>%
  summarize(mean = mean(mahalanobis_distance), median = median(mahalanobis_distance))

# Set the theme
theme_set(theme_ridges())

# Create the density ridge plot with four colors and mean/median lines
ggplot(data_long, aes(x = mahalanobis_distance, y = Clinical_Cohorts)) +
  geom_density_ridges(aes(fill = Clinical_Cohorts), alpha = 0.8) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800", "#FC4E07", "#7CFC00")) +
  geom_vline(data = summary_stats, aes(xintercept = mean, color = Clinical_Cohorts), linetype = "dashed", size = 0.5) +
  geom_vline(data = summary_stats, aes(xintercept = median, color = Clinical_Cohorts), linetype = "dotted", size = 0.5) +
  labs(title = "Density Ridge Plot with Mean and Median",
       x = "Mahalanobis Distance",
       y = "Clinical Cohorts",
       fill = "Clinical Cohorts") +
  theme(legend.position = "right")


# data(iris)
# head(iris)

# theme_set(theme_ridges())

# # Create the density ridge plot
# ggplot(iris, aes(x = Sepal.Length, y = Species)) +
#   geom_density_ridges(aes(fill = Species)) +
#   scale_fill_manual(values = c("#00AFBB", "#E7B800", "#FC4E07"))
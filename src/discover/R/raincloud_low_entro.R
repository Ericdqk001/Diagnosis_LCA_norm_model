# Install and load required packages
# install.packages("bit")
library(ggpubr)
library(ggplot2)
library(dplyr)
library(readr)

# Define the color palette for consistent cohort colors
cohort_colors <- c("HCs" = "#1f77b4", "PI" = "#ff7f0e", "PE" = "#2ca02c", "HD" = "#d62728")

# Load U test results
u_test_results <- read_csv("src/discover/results/low_entropy/U_test_results/recon_dev_U_test_results.csv")

# Function to convert FDR p-values to star annotations
fdr_to_stars <- function(p_value) {
  if (p_value < 0.001) return("***")
  else if (p_value < 0.01) return("**")
  else if (p_value < 0.05) return("*")
  else return("")
}

# Function to add statistical annotations to the plots
add_stat_annotations <- function(data, u_test_results, feature_name, plot_title) {

  # Recode cohort labels
  data <- data %>%
    mutate(cohort = case_when(
      low_symp_test_subs == 1 ~ "HCs",
      inter_test_subs == 1 ~ "PI",
      exter_test_subs == 1 ~ "PE",
      high_test_subs == 1 ~ "HD"
    ))

  # Ensure cohort order is HCs, PI, PE, HD
  data$cohort <- factor(data$cohort, levels = c("HCs", "PI", "PE", "HD"))

  # Filter U test results for the specific feature (Cortical Thickness, Volume, or Surface Area)
  feature_results <- u_test_results %>%
    filter(Feature == feature_name)

  # Get FDR p-values and their corresponding star annotations
  inter_fdr_star <- fdr_to_stars(feature_results$FDR_p_value[feature_results$Cohort == "inter_test"])
  exter_fdr_star <- fdr_to_stars(feature_results$FDR_p_value[feature_results$Cohort == "exter_test"])
  high_fdr_star <- fdr_to_stars(feature_results$FDR_p_value[feature_results$Cohort == "high_test"])

  # Calculate the maximum value for each cohort for star placement
  max_stats <- data %>%
    group_by(cohort) %>%
    summarise(y_max = max(reconstruction_deviation, na.rm = TRUE)) %>%
    mutate(y_max = y_max * 1.05)  # Add a small offset for better visibility of stars

  # Generate plot
  p <- ggplot(data, aes(x = cohort, y = reconstruction_deviation, fill = cohort)) + 
    ggdist::stat_halfeye(adjust = .5, width = .7, .width = 0, justification = -.2, point_colour = NA) + 
    geom_boxplot(width = .2, outlier.shape = NA) + 
    geom_jitter(width = .05, alpha = .3) + 
    xlab("Cohort") +
    ylab("Whole cortex deviation") +
    ggtitle(plot_title) +
    theme_classic(base_size = 16) +
    scale_fill_manual(values = cohort_colors) +
    annotate("text", x = 2, y = max_stats$y_max[max_stats$cohort == "PI"], label = inter_fdr_star, size = 8, color = "black") +  # PI vs HC
    annotate("text", x = 3, y = max_stats$y_max[max_stats$cohort == "PE"], label = exter_fdr_star, size = 8, color = "black") +  # PE vs HC
    annotate("text", x = 4, y = max_stats$y_max[max_stats$cohort == "HD"], label = high_fdr_star, size = 8, color = "black")  # HD vs HC

  return(p)
}

# Read the datasets
data_thickness <- read.csv("src/discover/results/output_data_low_entropy/t1w_cortical_thickness_rois_output_data_with_dev.csv")
data_volume <- read.csv("src/discover/results/output_data_low_entropy/t1w_cortical_volume_rois_output_data_with_dev.csv")
data_surface_area <- read.csv("src/discover/results/output_data_low_entropy/t1w_cortical_surface_area_rois_output_data_with_dev.csv")

# Generate plots with statistical annotations
plot_thickness <- add_stat_annotations(data_thickness, u_test_results, "Cortical Thickness", "")
plot_volume <- add_stat_annotations(data_volume, u_test_results, "Cortical Volume", "")
plot_surface_area <- add_stat_annotations(data_surface_area, u_test_results, "Cortical Surface Area", "")

# Save each individual plot with statistical annotations
ggsave("src/discover/R/images/cortical_thickness_distribution_with_stats.png", plot = plot_thickness, width = 12, height = 8, dpi = 300)
ggsave("src/discover/R/images/cortical_volume_distribution_with_stats.png", plot = plot_volume, width = 12, height = 8, dpi = 300)
ggsave("src/discover/R/images/cortical_surface_area_distribution_with_stats.png", plot = plot_surface_area, width = 12, height = 8, dpi = 300)


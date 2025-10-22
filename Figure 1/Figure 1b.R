# Load necessary libraries
library(MatchIt)
library(readxl)
library(dplyr)
library(openxlsx)

# Read the data from the Excel file
file_path <- "matchit_rtx.xlsx"  # Update with your correct file path if needed
data <- read_excel(file_path)
data$treat <- ifelse(data$group == "SSc", 1, 0)

# Check the data structure
str(data)

# Convert group to a factor if it's not already
data$group <- as.factor(data$group)

# Perform matching
matched_data <- matchit(treat ~ sex + age, data = data, method = "nearest", ratio = 1)

# Plot the matching result
plot(matched_data, type = "jitter", interactive = FALSE)

# Get the matched dataset
matched_df <- match.data(matched_data)

# Save the matched dataset to a new Excel file
write.xlsx(matched_df, "matched_data.xlsx", rowNames = FALSE)

# Display the matching summary
summary(matched_data)


# Install and load necessary packages
if (!requireNamespace("UpSetR", quietly = TRUE)) {
  install.packages("UpSetR")
}
library(UpSetR)

# Also load the readxl package for reading Excel files
if (!requireNamespace("readxl", quietly = TRUE)) {
  install.packages("readxl")
}
library(readxl)

# Read the data from an Excel file (ensure the path to your Excel file is correct)
data <- read_excel("6models_upsetplot.xlsx")

# Convert each column to a set (list element), omitting NA values
data_list <- lapply(data, function(x) {
  set <- as.character(x)
#  print(set)  # Debug: Print the contents of each set
  set
})

# Generate the UpSet plot
upset(fromList(data_list), sets = names(data_list), keep.order = TRUE, main.bar.color = "#56B4E9",
      sets.bar.color = "#D55E00", matrix.color = "#009E73", order.by = "freq")

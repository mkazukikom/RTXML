# 必要なパッケージを読み込む
library(ComplexHeatmap)
library(circlize)
library(dplyr)

# データの読み込み
rtx_data <- read.csv("rtx_data_matched.csv")
decreased_genes <- read.csv("decreased_items_week24_hr_mann_matched.csv")

# 遺伝子リストに基づいてデータをフィルタリング
genes_of_interest <- decreased_genes$gene
data_filtered <- rtx_data %>%
  select(group, arm, responder, timepoint, all_of(genes_of_interest))

# group=HC, group=SSc & arm=RTX & responder=1, それ以外のカラムを準備
data_filtered$color_group <- case_when(
  data_filtered$group == "HC" ~ "HC",
  data_filtered$arm == "Placebo" & data_filtered$timepoint == "week0" ~ "SSc Placebo week0",
  data_filtered$arm == "Placebo" & data_filtered$timepoint == "week24" ~ "SSc Placebo week24",
  data_filtered$arm == "RTX" & data_filtered$responder == 0 & data_filtered$timepoint == "week0" ~ "SSc RTX LR week0",
  data_filtered$arm == "RTX" & data_filtered$responder == 0 & data_filtered$timepoint == "week24" ~ "SSc RTX LR week24",
  data_filtered$arm == "RTX" & data_filtered$responder == 1 & data_filtered$timepoint == "week0" ~ "SSc RTX HR week0",
  data_filtered$arm == "RTX" & data_filtered$responder == 1 & data_filtered$timepoint == "week24" ~ "SSc RTX HR week24",

)

# ヒートマップ用のデータ行列を準備
heatmap_data <- data_filtered %>%
  select(-group, -arm, -responder, -timepoint, -color_group)

# 列ごとに正規化
normalize_data <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
norm_data <- as.data.frame(lapply(heatmap_data, normalize_data))
#norm_data$color_group <- data_filtered$color_group
norm_data <- t(norm_data)
norm_data <- as.data.frame(norm_data)

# sample_colorsの計算を修正
disease_colors <- setNames(c("grey", "green", "darkgreen", "cyan", "darkcyan", "orange", "darkorange"),
                           c("HC", "Placebo week0", "SSc Placebo week24", "SSc RTX HR week0", "SSc RTX HR week24","SSc RTX LR week0", "SSc RTX LR week24"))
sample_colors <- sapply(norm_data$CLASS, function(x) disease_colors[x])

# ヒートマップの作成
Heatmap(as.matrix(norm_data), 
        column_split = data_filtered$color_group,
        col = c("purple", "yellow"),
        name = "Normalized serum levels",
        top_annotation = HeatmapAnnotation(CLASS = anno_block(gp = gpar(fill = disease_colors))),
        cluster_rows = TRUE, 
        cluster_columns = FALSE,
        show_row_names = TRUE,
        show_column_names = FALSE,
        column_title = "",
        row_title = "Antigens")

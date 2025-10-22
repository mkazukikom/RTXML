# 必要なライブラリの読み込み
library(ComplexHeatmap)
library(circlize)
library(Hmisc)
library(dplyr)

# データの読み込み
rtx_data <- read.csv("rtx_data_matched.csv", stringsAsFactors = FALSE)
rtx_trait <- read.csv("rtx_trait_matched.csv", stringsAsFactors = FALSE)

# 遺伝子データと臨床データのマージ
merged_data <- merge(rtx_data, rtx_trait, by = "id")

# CSVファイルの読み込み（ファイルパスは適宜修正）
df <- read.csv("receptor_antigens.csv")
# 指定された遺伝子とAge以降の全臨床情報に絞り込む
genes <- unique(df$gene)
# Age列の位置を特定し、それ以降の全ての臨床情報列を取得
age_col_index <- which(names(merged_data) == "Age")
clinical_info_columns <- names(merged_data)[age_col_index:length(names(merged_data))]
selected_columns <- c("id", genes, clinical_info_columns) # "id" を含めて選択

# 相関係数とP値の計算のため、id列を除外
data_for_analysis <- merged_data[1:90, selected_columns[-1]] # id列を除外
data_for_analysis <- data_for_analysis %>% select(-HR.y, -CD19.y, -CD20) # HR列を除外

# 相関係数とP値の計算
cor_matrix <- cor(data_for_analysis, use = "complete.obs")
p_matrix <- rcorr(as.matrix(data_for_analysis), type = "spearman")$P

# ヒートマップ用に相関行列を整形
# 遺伝子と臨床情報の相関のみを抽出
genes_cor <- cor_matrix[genes, ]
genes_clinical_cor <- genes_cor[, 7:42]

# P値に基づくアノテーション
pval_annotation <- matrix("", nrow = nrow(genes_clinical_cor), ncol = ncol(genes_clinical_cor))
genes_p_values <- p_matrix[genes, 7:42]
pval_annotation[genes_p_values < 0.001] <- "***"
pval_annotation[genes_p_values < 0.01 & genes_p_values >= 0.001] <- "**"
pval_annotation[genes_p_values < 0.05 & genes_p_values >= 0.01] <- "*"

# 相関係数ヒートマップのクラスタリングと描画
Heatmap(genes_clinical_cor, 
        name = "correlation", 
        #        right_annotation = rowAnnotation(foo = anno_text(genes, rot = 90, location  = -unit(1, "cm"))),
        column_title = "Clinical Information", 
        row_title = "Antigens",
        cluster_rows = FALSE, 
        cluster_columns = FALSE, 
        show_row_names = TRUE, 
        show_column_names = TRUE,
        column_names_side = "bottom",
        column_names_gp = gpar(fontsize = 10, fontface = "bold"),
        row_names_gp = gpar(fontsize = 10, fontface = "bold"),
        cell_fun = function(j, i, x, y, width, height, fill) {
          grid.text(pval_annotation[i, j], x, y, gp = gpar(fontsize = 10))
        })

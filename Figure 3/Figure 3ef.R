# 必要なライブラリの読み込み
library(tidyverse)
library(factoextra)
library(ggrepel)

# データの読み込み
rtx_data <- read.csv("rtx_data_matched.csv")
decreased_genes <- read.csv("decreased_items_week24_hr_mann_matched.csv")

# decreased_genesに記載されている遺伝子のデータをrtx_dataから抽出
# timepoint=week24のサンプルを除外
genes_of_interest <- intersect(names(rtx_data), decreased_genes$gene)

data_for_pca <- rtx_data %>%
  select(group, arm, responder, timepoint, all_of(genes_of_interest))

# 色分けのためのカテゴリー列を追加
data_for_pca <- data_for_pca %>%
  mutate(color_category = case_when(
    group == "HC" ~ "HC",
    arm == "RTX" & responder == 0 & timepoint == "week0" ~ "SSc RTX LR week0",
    arm == "RTX" & responder == 1 & timepoint == "week0" ~ "SSc RTX HR week0",
    arm == "RTX" & responder == 0 & timepoint == "week24" ~ "SSc RTX LR week24",
    arm == "RTX" & responder == 1 & timepoint == "week24" ~ "SSc RTX HR week24",
    arm == "Placebo" & timepoint == "week0" ~ "SSc Placebo week0",
    arm == "Placebo" & timepoint == "week24" ~ "SSc Placebo week24",
  ))

data_for_pca$color_category <- as.factor(data_for_pca$color_category)

# PCA用のデータを準備（カテゴリー列を除外）
pca_data <- select(data_for_pca, -c(group, arm, responder, timepoint, color_category))

# PCAの実施
pca_result <- prcomp(pca_data, scale. = TRUE)

# --- 修正点1: プロット用のデータフレームを一つにまとめる ---
# PCAの結果とカテゴリ情報を結合
plot_data <- data.frame(pca_result$x,
                        color_category = data_for_pca$color_category)


# --- 修正点2 & 3: ggplotのコードを修正 ---
# `aes()`内は列名のみを記述し、scale_color_manualを使い、valuesの名前を正確に合わせる
p <- ggplot(plot_data, aes(x = PC1, y = PC2, color = color_category)) +
  geom_point() +
  scale_color_manual(
    values = c(
      "HC" = "grey",
      "SSc Placebo week0" = "green",
      "SSc Placebo week24" = "darkgreen",
      "SSc RTX LR week0" = "orange",
      "SSc RTX LR week24" = "darkorange",
      "SSc RTX HR week0" = "cyan",
      "SSc RTX HR week24" = "darkcyan"
    )
  ) +
  theme_minimal() +
  theme(legend.position = "right") + # 凡例を表示して色を確認
  labs(title = "PCA", x = "PC1", y = "PC2", color = "Group")

# マージナルヒストグラムを追加
ggExtra::ggMarginal(p,
                    type = "density", 
                    margins = "both", 
                    size = 3,
                    groupColour = TRUE,
                    groupFill = TRUE)

# PCAの負荷量をデータフレームに変換
loadings <- as.data.frame(pca_result$rotation[, 1:2])  # PC1とPC2の負荷量
colnames(loadings) <- c("PC1", "PC2")
loadings$Gene <- rownames(loadings)

# PC1とPC2の負荷量のグラフを表示（CCR8とMC1Rを赤くハイライト）
ggplot(loadings, aes(x = PC1, y = PC2, label = Gene)) +
  # geom_textにcolorのaestheticを追加
  geom_text(aes(color = Gene %in% c("CCR8", "NPFFR2", "P2RY8", "MC1R", "HTR1B", "FPR1")), vjust = 1.5, size = 3, show.legend = FALSE) +
  # scale_color_manualで色を定義 (TRUE=赤, FALSE=黒)
  scale_color_manual(values = c("TRUE" = "red", "FALSE" = "grey50")) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  theme_minimal() +
  labs(title = "Loadings Plot for PC1 and PC2",
       x = "PC1 Loadings",
       y = "PC2 Loadings") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

highlight_genes <- c("CCR8", "NPFFR2", "P2RY8", "MC1R", "HTR1B", "FPR1")
loadings$highlight <- loadings$Gene %in% highlight_genes

ggplot(loadings, aes(x = PC1, y = PC2)) +

  # 強調ラベル
  geom_text_repel(data = subset(loadings, highlight),
                  aes(label = Gene),
                  color = "red",
                  size = 4.5,
                  fontface = "bold") +
  # 通常ラベル
  geom_text(data = subset(loadings, !highlight),
                  aes(label = Gene),
                  color = "grey60",
                  size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey70") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey70") +
  theme_minimal() +
  labs(title = "Loadings Plot for PC1 and PC2",
       x = "PC1 Loadings",
       y = "PC2 Loadings") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# PC1の負荷量を棒グラフで表示
ggplot(loadings, aes(x = reorder(Gene, PC1), y = PC1)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Loadings for PC1", x = "Targeted Autoantigens", y = "Loading")

# 赤くしたい遺伝子のリスト
highlight_genes <- c("CCR8", "NPFFR2", "P2RY8", "MC1R", "HTR1B", "FPR1")

# 'highlight'列を追加して色分け
loadings$highlight <- ifelse(loadings$Gene %in% highlight_genes, "highlight", "normal")

# プロット
ggplot(loadings, aes(x = reorder(Gene, PC1), y = PC1, fill = highlight)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  scale_fill_manual(values = c("highlight" = "red", "normal" = "gray")) +
  labs(title = "Loadings for PC1", x = "Targeted Autoantigens", y = "Loading") +
  theme(legend.position = "none")  # 凡例を消す場合

# PC2の負荷量を棒グラフで表示
ggplot(loadings, aes(x = reorder(Gene, PC2), y = PC2)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Loadings for PC2", x = "Targeted Autoantigens", y = "Loading")

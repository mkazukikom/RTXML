library(dplyr)
library(purrr)
library(broom)
library(ggplot2)

# データの読み込み
data <- read.csv("rtx_data_matched.csv")
gene_list <- read.csv("increased_items_ssc_mann_matched.csv")

# significant_genesのリスト
significant_genes <- gene_list$Gene

# arm=RTXのデータをフィルタリング
data_filtered <- data %>%
  filter(arm == "RTX", timepoint == "week0") %>%
  select(id, responder, all_of(significant_genes))

# week0とweek24のデータを分ける
data_LR <- data_filtered %>% filter(responder == 0)
data_HR <- data_filtered %>% filter(responder == 1)

# Mann-Whitney U検定を実行
wilcox_results <- map_dfr(significant_genes, ~wilcox.test(
  x = data_LR[[.x]],
  y = data_HR[[.x]],
  paired = FALSE,
  exact = FALSE
) %>% broom::tidy() %>% mutate(gene = .x))

# Log2 Fold Changeを計算
log2_fold_changes <- map_dbl(significant_genes, ~log2(
  mean(data_HR[[.x]], na.rm = TRUE) /
    mean(data_LR[[.x]], na.rm = TRUE)
))

wilcox_results$log2FoldChange <- log2_fold_changes

# ハイライト条件の追加
wilcox_results$Highlight <- wilcox_results$p.value < 0.05 & wilcox_results$log2FoldChange > 1

# ボルケーノプロットの作成
ggplot(wilcox_results, aes(x = log2FoldChange, y = -log10(p.value))) +
  geom_point(aes(color = Highlight), alpha = 0.5) +
  scale_color_manual(values = c("FALSE" = "orange", "TRUE" = "cyan")) +
  labs(title = "LR vs HR",
       x = "Log2 Fold Change",
       y = "-Log10(P-Value)") +
  xlim(c(-7.5, 7.5)) +
  ylim(c(0, 6)) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "blue") +
  geom_vline(xintercept = 1, linetype = "dashed", color = "blue") +
  theme_minimal() +
  theme(legend.position = "none")

# ハイライトしたい遺伝子を指定
label_genes <- c("CCR8", "NPFFR2", "P2RY8", "MC1R", "HTR1B", "FPR1")

# ハイライト用の列を追加
wilcox_results$Label <- wilcox_results$gene %in% label_genes

# Volcano plotの作成
ggplot(wilcox_results, aes(x = log2FoldChange, y = -log10(p.value))) +
  # geom_pointでハイライトする点をシアン色で表示
  geom_point(aes(color = Highlight), alpha = 0.6) +
  scale_color_manual(values = c("FALSE" = "orange", "TRUE" = "cyan")) +
  
  # geom_text_repelでハイライトする点にのみラベルを太字で表示
  geom_text_repel(
    data = subset(wilcox_results, Label == TRUE), # ハイライトするデータのみを対象
    aes(label = gene),
#    fontface = "bold", # フォントを太字に
    color = "black",
    box.padding = 0.5,
    point.padding = 0.5,
    segment.color = 'grey50'
  ) +
  
  # 閾値の線
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "blue") +
  geom_vline(xintercept = 1, linetype = "dashed", color = "blue") +
  
  # ラベルとテーマ
  labs(title = "LR vs HR",
       x = "Log2 Fold Change",
       y = "-Log10(P-Value)") +
  xlim(c(-7.5, 7.5)) +
  ylim(c(0, 6)) +
  theme_minimal() +
  theme(legend.position = "none")

# 条件を満たす遺伝子を抽出し、CSVに出力
filtered_genes_to_export <- wilcox_results %>%
  filter(log2FoldChange > 1 & p.value < 0.05)

write.csv(filtered_genes_to_export, "increased_items_hr_mann_matched.csv", row.names = FALSE)

library(dplyr)
library(ggplot2)
library(ggrepel)

# データの読み込み
data <- read.csv("rtx_data_matched.csv")

# A4GALT以降の列で最大値が1.5625未満のものを除外
filtered_genes <- names(data)[grep("A4GALT", names(data)):ncol(data)]
filtered_genes <- filtered_genes[sapply(data[, filtered_genes], max, na.rm = TRUE) >= 1.5625]

# HCとSSc週0データを選択し、フィルタリングされた列のみを含む
hc_data <- filter(data, group == 'HC')[, c("group", "timepoint", filtered_genes)]
ssc_week0_data <- filter(data, group == 'SSc', timepoint == 'week0')[, c("group", "timepoint", filtered_genes)]

# ノンパラメトリック検定を実施し、p値と効果サイズを計算
results <- lapply(filtered_genes, function(gene) {
  hc_values <- hc_data[[gene]]
  ssc_values <- ssc_week0_data[[gene]]
  
  # Wilcoxon rank-sum test
  test_result <- wilcox.test(ssc_values, hc_values, alternative = "two.sided", exact = FALSE)
  
  data.frame(
    Gene = gene,
    FoldChange = mean(ssc_values, na.rm = TRUE)/mean(hc_values, na.rm = TRUE),
    PValue = test_result$p.value
  )
}) %>% bind_rows()

# -log10(PValue)を計算
results <- results %>%
  mutate(NegLog10PValue = -log10(PValue)) %>%
  mutate(Log2FoldChange = log2(FoldChange))

# 指定された条件を満たす遺伝子のハイライト
results$Highlight <- results$PValue < 0.05 & results$Log2FoldChange > 1

# Volcano plotの描画
ggplot(results, aes(x = Log2FoldChange, y = NegLog10PValue)) +
  geom_point(aes(color = Highlight), alpha = 0.5) +
  scale_color_manual(values = c("FALSE" = "grey", "TRUE" = "magenta")) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "blue") +
  geom_vline(xintercept = 1, linetype = "dashed", color = "blue") +
  labs(title = "HC vs SSc", x = "Log2 Fold Change", y = "-log10(P-value)") +
  theme_minimal() +
  theme(legend.position = "none")

# ハイライトしたい遺伝子を指定
label_genes <- c("CCR8", "NPFFR2", "P2RY8", "MC1R", "HTR1B", "FPR1")

# ハイライト用の列を追加
results$Label <- results$Gene %in% label_genes

# Volcano plotの描画
ggplot(results, aes(x = Log2FoldChange, y = NegLog10PValue)) +
  # geom_pointでハイライトする点をマゼンタ色で表示
  geom_point(aes(color = Highlight), alpha = 0.6, size = 2) +
  scale_color_manual(values = c("FALSE" = "grey", "TRUE" = "magenta")) +
  
  # geom_text_repelでハイライトする点にのみラベルを太字で表示
  geom_text_repel(
    data = subset(results, Label == TRUE), # ハイライトするデータのみを対象
    aes(label = Gene),
#    fontface = "bold", # フォントを太字に
    color = "black",
    box.padding = 0.5,
    point.padding = 0.5,
    segment.color = 'black',
    min.segment.length = 0
  ) +
  labs(title = "min.segment.length = 0") +
  
  # 閾値の線
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "blue") +
  geom_vline(xintercept = 1, linetype = "dashed", color = "blue") +
  
  # ラベルとテーマ
  labs(title = "HC vs SSc", x = "Log2 Fold Change", y = "-log10(P-value)") +
  theme_minimal() +
  theme(legend.position = "none")

# 条件を満たす遺伝子を抽出し、CSVに出力
filtered_genes_to_export <- results %>%
  filter(Log2FoldChange > 1 & PValue < 0.05)

write.csv(filtered_genes_to_export, "increased_items_ssc_mann_matched.csv", row.names = FALSE)

# -log10(PValue)とlog2(FoldChange)の計算、FDR補正の追加
results <- results %>%
  mutate(
    Log2FoldChange = log2(FoldChange),
    PAdj = p.adjust(PValue, method = "BH"),
    NegLog10PValue = -log10(PValue),
    NegLog10PAdj = -log10(PAdj),
    Highlight = PAdj < 0.1 & Log2FoldChange > 1
  )

filtered_genes_to_export <- results %>%
  filter(Log2FoldChange > 1 & PAdj < 0.1)

write.csv(filtered_genes_to_export, "increased_items_ssc_mann_matched_FDR.csv", row.names = FALSE)

# データの読み込み
data <- read.csv("rtx_data_matched_early.csv")

# A4GALT以降の列で最大値が1.5625未満のものを除外
filtered_genes <- names(data)[grep("A4GALT", names(data)):ncol(data)]
filtered_genes <- filtered_genes[sapply(data[, filtered_genes], max, na.rm = TRUE) >= 1.5625]

# HCとSSc週0データを選択し、フィルタリングされた列のみを含む
hc_data <- filter(data, group == 'HC')[, c("group", "timepoint", filtered_genes)]
ssc_week0_data <- filter(data, group == 'SSc', timepoint == 'week0')[, c("group", "timepoint", filtered_genes)]

# ノンパラメトリック検定を実施し、p値と効果サイズを計算
results <- lapply(filtered_genes, function(gene) {
  hc_values <- hc_data[[gene]]
  ssc_values <- ssc_week0_data[[gene]]
  
  # Wilcoxon rank-sum test
  test_result <- wilcox.test(ssc_values, hc_values, alternative = "two.sided", exact = FALSE)
  
  data.frame(
    Gene = gene,
    FoldChange = mean(ssc_values, na.rm = TRUE)/mean(hc_values, na.rm = TRUE),
    PValue = test_result$p.value
  )
}) %>% bind_rows()

# -log10(PValue)とlog2(FoldChange)の計算、FDR補正の追加
results <- results %>%
  mutate(
    Log2FoldChange = log2(FoldChange),
    PAdj = p.adjust(PValue, method = "BH"),
    NegLog10PValue = -log10(PValue),
    NegLog10PAdj = -log10(PAdj),
    Highlight = PAdj < 0.1 & Log2FoldChange > 1
  )

filtered_genes_to_export <- results %>%
  filter(Log2FoldChange > 1 & PAdj < 0.1)

write.csv(filtered_genes_to_export, "increased_items_ssc_mann_matched_early_FDR.csv", row.names = FALSE)

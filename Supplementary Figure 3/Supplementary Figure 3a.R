# 必要なライブラリの読み込み
library(ggplot2)
library(dplyr)
library(gridExtra)
library(ggpubr)

# データの読み込み
rtx_data <- read.csv("rtx_data_matched.csv", stringsAsFactors = FALSE)
rtx_trait <- read.csv("rtx_trait_matched.csv", stringsAsFactors = FALSE)

# データのマージ
merged_data <- merge(rtx_data, rtx_trait, by = "id")

# HAQに関連するカラム名
haq_vars <- c("mRSS", "HAQ", paste0("HAQ", 1:8))

# 有効な散布図を格納するリスト
plot_list <- list()

# 各HAQ項目について CCR8との散布図を作成
for (var in haq_vars) {
  scatter_data <- merged_data %>%
    select(CCR8, !!sym(var)) %>%
    rename(HAQ_Item = !!sym(var)) %>%
    na.omit()
  
  # Spearman相関の計算
  cor_test <- cor.test(scatter_data$CCR8, scatter_data$HAQ_Item, method = "spearman")
  rho <- round(cor_test$estimate, 2)
  pval <- cor_test$p.value
  
  # P値のカテゴリ分け表記
  pval_label <- case_when(
    pval < 0.001 ~ "P < 0.001",
    pval < 0.01  ~ "P < 0.01",
    pval < 0.05  ~ "P < 0.05",
    TRUE         ~ "ns"
  )
  
  p <- ggplot(scatter_data, aes(x = CCR8, y = HAQ_Item)) +
    geom_point(size = 2, alpha = 0.8) +
    geom_smooth(method = "lm", se = TRUE, color = "steelblue") +
    labs(subtitle = paste0("R = ", rho, ", ", pval_label), x = "anti-CCR8 [AU]", y = var) +
    theme_minimal(base_size = 12)
  
  plot_list[[length(plot_list) + 1]] <- p
}

# 要素数が10になるように空プロットを追加（2x5に整える）
while (length(plot_list) < 10) {
  plot_list[[length(plot_list) + 1]] <- ggplot() + theme_void()
}

# グリッド表示（2行5列）
grid.arrange(grobs = plot_list, ncol = 5, nrow = 2)


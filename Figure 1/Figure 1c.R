# ライブラリの読み込み
library(tidyverse)
library(ggpubr)
library(patchwork)

# データの読み込み
df1 <- read.csv("rtx_data_matched.csv")
df2 <- read.csv("rtx_trait_matched.csv")

# IDでマージ
merged_df <- inner_join(df1, df2, by = "id")

# P値ラベルの関数
format_pval <- function(p) {
  if (p < 0.001) {
    "P < 0.001"
  } else if (p < 0.01) {
    "P < 0.01"
  } else if (p < 0.05) {
    "P < 0.05"
  } else {
    "ns"
  }
}

# 相関解析と図の作成関数
analyze_and_plot <- function(data, var1, var2, label) {
  df <- data %>% select(all_of(c(var1, var2))) %>% drop_na()
  
  # スピアマン相関
  test <- cor.test(df[[var1]], df[[var2]], method = "spearman", exact = FALSE)
  rho <- round(test$estimate, 2)
  pval_label <- format_pval(test$p.value)
  
  # 相関係数・p値・信頼区間（bootstrapping）
  boot <- cor.test(df[[var1]], df[[var2]], method = "spearman", conf.level = 0.95)
  
  # 結果出力
  cat(paste0(label, "\n"))
  cat(paste0("Spearman ρ = ", rho, ", ", pval_label, "\n"))
  
  # 散布図 + 回帰線
  p <- ggscatter(df, x = var1, y = var2,
                 add = "reg.line", conf.int = TRUE,
                 add.params = list(color = "blue", fill = "lightgray")) +
    annotate("text", x = min(df[[var1]], na.rm = TRUE),
             y = max(df[[var2]], na.rm = TRUE) * 0.95,
             label = paste0("ρ = ", rho, ", ", pval_label),
             hjust = 0, size = 4) +
    labs(title = label, x = "ELISA [U/mL]", y = "WPA [AU]") +
    theme_minimal()
  
  return(p)
}

# 各比較ペアに対して解析＆図
p1 <- analyze_and_plot(merged_df, "ATA", "TOP1", "ATA")
p2 <- analyze_and_plot(merged_df, "ACA", "CENPB", "ACA")
p3 <- analyze_and_plot(merged_df, "ARA", "POLR3A_D", "ARA")

# 横一列に並べて表示
(p1 | p2 | p3) + plot_layout(guides = 'collect')

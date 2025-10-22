# 必要なパッケージを読み込む
library(ggplot2)

# データを読み込む
data <- read.csv("rtx_trait.csv")

# ΔmRSSが-6.5より大きい場合はRTX LR、それ以外はRTX HRと分類
data$group <- ifelse(data$arm == "RTX" & data$ΔmRSS > -6.5, "RTX LR", 
                        ifelse(data$arm == "RTX", "RTX HR", data$arm))

# カラーを定義
colors <- c("RTX HR" = "cyan", "RTX LR" = "orange", "Placebo" = "green")

# ヒストグラムをプロット
p <- ggplot(data, aes(x = ΔmRSS, fill = group)) +
  geom_histogram(binwidth = 0.75, alpha = 0.75, position = "stack") +
  scale_fill_manual(values = colors) +
  geom_vline(xintercept = -6.5, color = "black", linetype = "dashed") +
  labs(x = "ΔmRSS", y = "Frequency") +
  theme_minimal()

# X軸を反転
p + scale_x_reverse()

# プロットを表示
print(p)

library(dplyr)
library(ggplot2)

# データの読み込み
data <- read.csv("rtx_data_matched.csv")  # 適切なファイルパスに置き換えてください

# A4GALT以降の列の総和を行ごとに計算
data$total_sum <- rowSums(data[, grep("A4GALT", names(data)):ncol(data)])

# 'group'列が空欄ではないデータのみをフィルタリング
filtered_data <- data %>% 
  filter(group != "")

# 'group'列の値を因子として設定し、レベルを指定
filtered_data$group <- factor(filtered_data$group, levels = c("HC", "SSc"))

# 箱ひげ図を描画
ggplot(filtered_data, aes(x = group, y = total_sum, fill = group)) +
  geom_boxplot() +
  scale_fill_manual(values = c("HC" = "grey", "SSc" = "magenta")) +
  labs(x = "", y = "SAL [AU]") +
  ylim(0,15000) +
  theme_minimal() +
  theme(text = element_text(size = 12))

# Wilcoxon rank sum testを実施
wilcoxon_test_result <- wilcox.test(total_sum ~ group, data = filtered_data)

# 結果の出力
print(wilcoxon_test_result)

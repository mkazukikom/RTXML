# 必要なパッケージの読み込み
library(ggplot2)
library(dplyr)
library(readr)

# データの読み込み
data <- read.csv("rtx_data_matched.csv")

# A4GALT列以降の総和を計算
data$total_sum <- rowSums(data[, which(colnames(data) == "A4GALT"):ncol(data)])

# 年代カテゴリを定義
data$age_group <- cut(data$age,
                      breaks = c(-Inf, 39, 59, Inf),
                      labels = c("<40", "40-59", "60<"))

# 条件に基づく新しい変数を生成
data <- data %>%
  mutate(
    conditions = case_when(
      group == "HC" ~ "HC",
      arm == "Placebo" & timepoint == "week0" ~ "Placebo week0",
      arm == "Placebo" & timepoint == "week24" ~ "Placebo week24",
      arm == "RTX" & timepoint == "week0" & responder == 0 ~ "RTX LR week0",
      arm == "RTX" & timepoint == "week24" & responder == 0 ~ "RTX LR week24",
      arm == "RTX" & timepoint == "week0" & responder == 1 ~ "RTX HR week0",
      arm == "RTX" & timepoint == "week24" & responder == 1 ~ "RTX HR week24"
    )
  )

# プロットの順序を指定
condition_labels <- c("HC", "Placebo week0", "Placebo week24", 
                      "RTX HR week0", "RTX HR week24",
                      "RTX LR week0", "RTX LR week24"
                      )

# 条件に応じて色を指定
colors <- c("HC" = "gray", "Placebo week0" = "green", 
                                 "Placebo week24" = "lightgreen", "RTX LR week0" = "orange", 
                                 "RTX LR week24" = "darkorange", "RTX HR week0" = "cyan", 
                                 "RTX HR week24" = "darkcyan")

# 年代別の箱髭図を描画し、色を適用
ggplot(data, aes(x = factor(conditions, levels=condition_labels), y = total_sum, fill = conditions)) +
  geom_boxplot() +
  scale_fill_manual(values = colors) +
  facet_wrap(~ age_group, scales = "free_x") +

  labs(x = "",
       y = "SAL [AU]",
       fill = "Condition") +
  theme_minimal() +
  theme(axis.text.x = element_blank())

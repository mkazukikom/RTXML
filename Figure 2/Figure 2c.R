library(ggplot2)
library(dplyr)
library(tidyr)
library(stats)

# データの読み込み
data <- read.csv("rtx_data_matched.csv")

# A4GALT以降の列の総和を計算
data$total_sum <- rowSums(data[, which(colnames(data) == "A4GALT"):ncol(data)])

# 'arm'と'timepoint'の組み合わせで新しい列を作成
data$conditions <- paste(data$arm, data$timepoint, sep=" ")

# 条件に基づく新しい変数を生成
data <- data %>%
  mutate(
    conditions = case_when(
      arm == "Placebo" & timepoint == "week0" ~ "Placebo week0",
      arm == "Placebo" & timepoint == "week24" ~ "Placebo week24",
      arm == "RTX" & timepoint == "week0" & responder == 0 ~ "RTX LR week0",
      arm == "RTX" & timepoint == "week24" & responder == 0 ~ "RTX LR week24",
      arm == "RTX" & timepoint == "week0" & responder == 1 ~ "RTX HR week0",
      arm == "RTX" & timepoint == "week24" & responder == 1 ~ "RTX HR week24"
    )
  )

# プロットの順序を指定
condition_labels <- c("Placebo week0", "Placebo week24", 
                      "RTX HR week0", "RTX HR week24",
                      "RTX LR week0", "RTX LR week24"
                      )

# 箱ひげ図をプロット
ggplot(data[1:90,], aes(x=factor(conditions, levels=condition_labels), y=total_sum, fill=conditions)) +
  geom_boxplot() +
  scale_fill_manual(values=c("Placebo week0"="green", "Placebo week24" = "darkgreen",
                             "RTX LR week0"="orange", "RTX LR week24"="darkorange",
                             "RTX HR week0"="cyan", "RTX HR week24"="darkcyan")) +
  labs(x="Group", y="SAL [AU]") +
  theme_minimal() +
  theme(text = element_text(size=14), axis.text.x = element_text(angle=45, hjust=1))

# 対応のある検定 (paired t-test)
wilcox.test(data[data$conditions == "RTX LR week0", "total_sum"], data[data$conditions == "RTX LR week24", "total_sum"], paired = TRUE)
wilcox.test(data[data$conditions == "RTX HR week0", "total_sum"], data[data$conditions == "RTX HR week24", "total_sum"], paired = TRUE)

# 対応のない検定 (unpaired t-test)
wilcox.test(data[data$conditions == "RTX HR week0", "total_sum"], data[data$conditions == "RTX LR week0", "total_sum"], paired = FALSE)

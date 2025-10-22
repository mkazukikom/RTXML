library(ggplot2)
library(dplyr)
library(tidyr)
library(stats)

# データの読み込み
data <- read.csv("rtx_trait_matched.csv")

# 'arm'と'timepoint'の組み合わせで新しい列を作成
data$conditions <- paste(data$arm, data$timepoint, sep=" ")

# 条件に基づく新しい変数を生成
data <- data %>%
  mutate(
    conditions = case_when(
      arm == "Placebo" & timepoint == "week0" ~ "Placebo week0",
      arm == "Placebo" & timepoint == "week24" ~ "Placebo week24",
      arm == "RTX" & timepoint == "week0" & HR == 0 ~ "RTX LR week0",
      arm == "RTX" & timepoint == "week24" & HR == 0 ~ "RTX LR week24",
      arm == "RTX" & timepoint == "week0" & HR == 1 ~ "RTX HR week0",
      arm == "RTX" & timepoint == "week24" & HR == 1 ~ "RTX HR week24"
    )
  )

# プロットの順序を指定
condition_labels <- c("Placebo week0", "Placebo week24", 
                      "RTX HR week0", "RTX HR week24",
                      "RTX LR week0", "RTX LR week24"
                      )

# 箱ひげ図をプロット
ggplot(data, aes(x=factor(conditions, levels=condition_labels), y=ATA, fill=conditions)) +
  geom_boxplot() +
  scale_fill_manual(values=c("Placebo week0"="green", "Placebo week24" = "darkgreen",
                             "RTX LR week0"="orange", "RTX LR week24"="darkorange",
                             "RTX HR week0"="cyan", "RTX HR week24"="darkcyan")) +
  labs(title = "ATA", x="Group", y="Serum level [U/mL]") +
  theme_minimal() +
  theme(text = element_text(size=14), axis.text.x = element_text(angle=45, hjust=1))

# 対応のある検定
wilcox.test(data[data$conditions == "RTX LR week0", "ATA"], data[data$conditions == "RTX LR week24", "ATA"], paired = TRUE)
wilcox.test(data[data$conditions == "RTX HR week0", "ATA"], data[data$conditions == "RTX HR week24", "ATA"], paired = TRUE)

# 対応のない検定
wilcox.test(data[data$conditions == "RTX HR week0", "ATA"], data[data$conditions == "RTX LR week0", "ATA"], paired = FALSE)

# 箱ひげ図をプロット
ggplot(data, aes(x=factor(conditions, levels=condition_labels), y=ACA, fill=conditions)) +
  geom_boxplot() +
  scale_fill_manual(values=c("HC"="grey", "Placebo week0"="green", "Placebo week24" = "darkgreen",
                             "RTX LR week0"="orange", "RTX LR week24"="darkorange",
                             "RTX HR week0"="cyan", "RTX HR week24"="darkcyan")) +
  labs(title = "ACA", x="Group", y="Serum level [U/mL]") +
  theme_minimal() +
  theme(text = element_text(size=14), axis.text.x = element_text(angle=45, hjust=1))

# 対応のある検定
wilcox.test(data[data$conditions == "RTX LR week0", "ACA"], data[data$conditions == "RTX LR week24", "ACA"], paired = TRUE)
wilcox.test(data[data$conditions == "RTX HR week0", "ACA"], data[data$conditions == "RTX HR week24", "ACA"], paired = TRUE)

# 対応のない検定
wilcox.test(data[data$conditions == "RTX HR week0", "ACA"], data[data$conditions == "RTX LR week0", "ACA"], paired = FALSE)

# 箱ひげ図をプロット
ggplot(data, aes(x=factor(conditions, levels=condition_labels), y=ARA, fill=conditions)) +
  geom_boxplot() +
  scale_fill_manual(values=c("HC"="grey", "Placebo week0"="green", "Placebo week24" = "darkgreen",
                             "RTX LR week0"="orange", "RTX LR week24"="darkorange",
                             "RTX HR week0"="cyan", "RTX HR week24"="darkcyan")) +
  labs(title = "ARA", x="Group", y="Serum level [U/mL]") +
  theme_minimal() +
  theme(text = element_text(size=14), axis.text.x = element_text(angle=45, hjust=1))

# 対応のある検定
wilcox.test(data[data$conditions == "RTX LR week0", "ARA"], data[data$conditions == "RTX LR week24", "ARA"], paired = TRUE)
wilcox.test(data[data$conditions == "RTX HR week0", "ARA"], data[data$conditions == "RTX HR week24", "ARA"], paired = TRUE)

# 対応のない検定
wilcox.test(data[data$conditions == "RTX HR week0", "ARA"], data[data$conditions == "RTX LR week0", "ARA"], paired = FALSE)

# 必要なパッケージの読み込み
library(ggplot2)
library(dplyr)

# データの読み込み（適切なパスに置き換えてください）
data <- read.csv("rtx_data_matched.csv")

# グループ分類の条件を満たす新しい列をデータに追加
data$group_category <- with(data, case_when(
  group == "HC" ~ "HC",
  arm == "Placebo" & timepoint == "week0" ~ "Placebo week0",
  arm == "Placebo" & timepoint == "week24" ~ "Placebo week24",
  arm == "RTX" & timepoint == "week0" & responder == 1 ~ "RTX HR week0",
  arm == "RTX" & timepoint == "week24" & responder == 1 ~ "RTX HR week24",
  arm == "RTX" & timepoint == "week0" & responder == 0 ~ "RTX LR week0",
  arm == "RTX" & timepoint == "week24" & responder == 0 ~ "RTX LR week24",
  TRUE ~ "Other"
))

# 'Other'カテゴリを除外
filtered_data <- filter(data, group_category != "Other")

# プロットの順序を指定
condition_labels <- c("HC", "Placebo week0", "Placebo week24", 
                      "RTX LR week0", "RTX LR week24",
                      "RTX HR week0", "RTX HR week24")

# CSVファイルの読み込み（ファイルパスは適宜修正）
df <- read.csv("receptor_antigens.csv")
# 指定された遺伝子とAge以降の全臨床情報に絞り込む
genes <- unique(df$gene)
# 各遺伝子に対する箱ひげ図を作成
plot_list <- list()

for (gene in genes) {
  p <- ggplot(filtered_data, aes_string(x = "group_category", y = gene, fill = "group_category")) +
    geom_boxplot() +
    scale_fill_manual(values=c("HC"="grey", "Placebo week0"="green", "Placebo week24" = "darkgreen",
                               "RTX LR week0"="orange", "RTX LR week24"="darkorange",
                               "RTX HR week0"="cyan", "RTX HR week24"="darkcyan")) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = gene, x = "", y = "Serum autoantibody levels [AU]") +
    theme(legend.position = "none")
  plot_list[[gene]] <- p
}

# プロットの表示（選択した方法に応じて調整）
gridExtra::grid.arrange(grobs = plot_list, ncol = 4)

# 必要なパッケージの読み込み
library(ggplot2)
library(dplyr)
library(readr)
library(tidyr)
library(gridExtra)

# データの読み込み
data <- read_csv("rtx_data_matched.csv")

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

# 指定されたタンパク質のリスト
proteins <- c("TOP1", "CENPA", "CENPB", "CENPC", "POLR3A_D", "POLR3C", "POLR1A", "POLR2A",
              "POP1", "RPP25", "FBL", "UBTF", "RNPC3", "SSSCA1", "EIF2B2",
              "DLAT", "DLST", "DBT", "PDHX", "COIL", "SNRNP70", "SNRPA", "SNRPC", "SNRPB2",
              "XRCC5", "XRCC6", "EXOSC10", "EXOSC9", "RUVBL1", "RUVBL2", "PSME3", "TRIM21(1-400)", "TROVE2", "SSB")

# グラフを作成する関数
plot_protein <- function(protein) {
  ggplot(data, aes(x = factor(conditions, levels=condition_labels), y = .data[[protein]])) +
    geom_boxplot(aes(fill = conditions)) +
    scale_fill_manual(values = c("HC" = "gray", "Placebo week0" = "green", 
                                 "Placebo week24" = "lightgreen", "RTX LR week0" = "orange", 
                                 "RTX LR week24" = "darkorange", "RTX HR week0" = "cyan", 
                                 "RTX HR week24" = "darkcyan")) +
    labs(title = protein, y = "Seum level [AU]", x = "") +
    theme_minimal() +
    theme(axis.text.x = element_blank()) +
    theme(legend.position = "none")
}

# 全タンパク質に対するグラフをリストに格納
plots <- lapply(proteins, plot_protein)

# グラフのリストをパネル表示
do.call(grid.arrange, c(plots, ncol = 6))


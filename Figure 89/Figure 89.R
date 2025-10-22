## RNA-seq Analysis for Differential Gene Expression (DESeq2)

# Load required libraries
library(DESeq2)
library(dplyr)
library(data.table)
library(pheatmap)
library(EnhancedVolcano)
library(clusterProfiler)
library(org.Mm.eg.db)
library(msigdbr)
library(enrichplot)
library(openxlsx)
library(ggplot2)
library(stringr)
library(scales)


# setwd("your/data/directory")

#Read count + TPM data
files <- c("ca1.csv", "ca4.csv", "ca5.csv",
           "cb1.csv", "cb2.csv", "cb5.csv",
           "cc1.csv", "cc2.csv", "cc3.csv")

count_list <- lapply(files, function(file) {
  df <- read.csv(file)
  df <- df[, c("id", "symbol", "count", "tpm")]
  colnames(df)[3] <- paste0("count_", gsub(".csv", "", file))
  colnames(df)[4] <- paste0("tpm_", gsub(".csv", "", file))  
  return(df)
})

gene_level_data <- Reduce(function(x, y) merge(x, y, by = c("id", "symbol"), all = TRUE), count_list)
gene_level_data_dt <- as.data.table(gene_level_data)

merged_data <- gene_level_data_dt %>%
  group_by(symbol) %>%
  summarise(
    ids = paste(unique(id), collapse = ";"),
    across(starts_with("count_"), sum, na.rm = TRUE),
    across(starts_with("tpm_"), sum, na.rm = TRUE)
  )

gene_level_data <- merged_data

# Extract count matrix
count_matrix <- as.data.frame(gene_level_data[, grep("^count_", colnames(gene_level_data))])
colnames(count_matrix) <- gsub("count_", "", colnames(count_matrix))
rownames(count_matrix) <- gene_level_data$symbol
# Extract TPM matrix
tpm_matrix <- as.data.frame(gene_level_data[, grep("^tpm_", colnames(gene_level_data))])
colnames(tpm_matrix) <- gsub("tpm_", "", colnames(tpm_matrix))
rownames(tpm_matrix) <- gene_level_data$symbol

# DESeq2 dataset
condition <- factor(rep(c("PBS", "BLM + anti-CCR8", "BLM + IgG"), each = 3), 
                    levels = c("PBS", "BLM + anti-CCR8", "BLM + IgG"))
colData <- data.frame(row.names = colnames(count_matrix), condition = condition)

dds <- DESeqDataSetFromMatrix(count_matrix, colData, design = ~ condition)
dds <- dds[rowSums(counts(dds)) > 10, ]
dds <- DESeq(dds)

# Pairwise comparisons
comparisons <- list(
  c("BLM + anti-CCR8", "PBS"),
  c("BLM + IgG", "PBS"),
  c("BLM + anti-CCR8", "BLM + IgG")
)
comparison_names <- c("BLM+antiCCR8_vs_PBS", "BLM+IgG_vs_PBS", "BLM+antiCCR8_vs_BLM+IgG")

for (i in seq_along(comparisons)) {
  cond1 <- comparisons[[i]][1]
  cond2 <- comparisons[[i]][2]
  comp_name <- comparison_names[i]
  
  res <- results(dds, contrast = c("condition", cond1, cond2))
  res_df <- as.data.frame(res)
  res_df$symbol <- rownames(res_df)
  res_df <- res_df[!is.na(res_df$padj), ]
  
  cond1_samples <- rownames(colData)[colData$condition == cond1]
  cond2_samples <- rownames(colData)[colData$condition == cond2]
  
  tpm_matrix_filtered <- tpm_matrix[rownames(tpm_matrix) %in% rownames(res_df), , drop = FALSE]
  tpm_matrix_filtered <- tpm_matrix_filtered[match(rownames(res_df), rownames(tpm_matrix_filtered)), , drop = FALSE]
  
  mean_cond1 <- rowMeans(tpm_matrix_filtered[, cond1_samples, drop = FALSE])
  mean_cond2 <- rowMeans(tpm_matrix_filtered[, cond2_samples, drop = FALSE])
  
  res_df[[paste0("mean_TPM_", cond1)]] <- mean_cond1
  res_df[[paste0("mean_TPM_", cond2)]] <- mean_cond2
  res_df$ratio <- mean_cond1 / mean_cond2
  res_df$log2ratio <- log2(res_df$ratio)
  
  res_final <- merge(gene_level_data[, c("symbol", "ids")], res_df, by = "symbol", all.y = TRUE)
  res_final$significant <- ifelse(res_final$padj < 0.05, "yes", "no")
  
  write.csv(res_final, paste0("DESeq2_", comp_name, ".csv"), row.names = FALSE)
}

# DEG Heatmap (Figure 5C)
deg_files <- paste0("DESeq2_", comparison_names, ".csv")
deg_lists <- lapply(deg_files, function(file){
  deg <- read.csv(file) %>% filter(padj < 0.05)
  deg$symbol
})

all_DEGs <- unique(unlist(deg_lists))
tpm_DEGs <- tpm_matrix[rownames(tpm_matrix) %in% all_DEGs, ]

annotation_col <- data.frame(
  condition = rep(c("PBS", "BLM + anti-CCR8", "BLM + IgG"), each = 3),
  row.names = colnames(tpm_DEGs)
)

pheatmap(log2(tpm_DEGs + 1), scale = "row", cluster_cols = FALSE,
         annotation_col = annotation_col, show_rownames = FALSE, fontsize_col = 12,
         main = "Heatmap of DEGs across conditions")


# Filtering TPM < 0.3 
for (i in seq_along(deg_files)){
  res <- read.csv(deg_files[i], check.names = FALSE)
  conds <- comparisons[[i]]
  tpm_cols <- paste0("mean_TPM_", conds)
  filtered_res <- res[!(res[[tpm_cols[1]]] < 0.3 & res[[tpm_cols[2]]] < 0.3), ]
  write.csv(filtered_res, paste0("Filtered_", deg_files[i]), row.names = FALSE)
}


# Volcano Plot Function (Figure 5D)
volcano_plot_padj <- function(input_csv, tpm_col1, tpm_col2, output_png, plot_title) {
  df <- read.csv(input_csv, check.names = FALSE)
  df <- df[!(df[[tpm_col1]] < 0.3 & df[[tpm_col2]] < 0.3), ]
  df <- df[!is.na(df$padj) & is.finite(df$log2FoldChange), ]
  df$log_padj <- pmin(-log10(df$padj), 55)
  df$log2FoldChange <- pmax(pmin(df$log2FoldChange, 15), -15)
  
  # Gene category based on padj & log2FC
  df$color <- "Not Significant"
  df$color[df$log2FoldChange > 1 & df$padj < 0.05] <- "Red"
  df$color[df$log2FoldChange < -1 & df$padj < 0.05] <- "Blue"
  
  # Gene counts
  num_red <- sum(df$color == "Red")
  num_blue <- sum(df$color == "Blue")
  
  # Plot
  p <- ggplot(df, aes(x = log2FoldChange, y = log_padj, color = color)) +
    geom_point(alpha = 0.8, size = 2.5) +
    scale_color_manual(values = c("Red" = "red", "Blue" = "blue", "Not Significant" = "gray")) +
    scale_x_continuous(breaks = pretty_breaks(n = 10), limits = c(-15, 15)) +
    scale_y_continuous(expand = c(0, 0), limits = c(0, 55)) +
    theme_classic() +
    ggtitle(plot_title) +
    labs(x = "Log2 Fold Change", y = "-Log10 Adjusted P-value", color = "Significance") +
    geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "black", linewidth = 0.8) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "black", linewidth = 0.8) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
      axis.title.x = element_text(size = 16, face = "bold"),
      axis.title.y = element_text(size = 16, face = "bold"),
      axis.text = element_text(size = 14),
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 16, face = "bold")
    ) +
    annotate("text", x = 14, y = 40, label = paste0(" ", num_red), color = "red", size = 6, fontface = "bold") +
    annotate("text", x = -14, y = 40, label = paste0(" ", num_blue), color = "blue", size = 6, fontface = "bold")
  
  ggsave(output_png, plot = p, width = 10, height = 8, dpi = 300)
}


volcano_plot_padj(
  input_csv = "DESeq2_BLM+antiCCR8_vs_PBS.csv",
  tpm_col1 = "mean_TPM_BLM + anti-CCR8",
  tpm_col2 = "mean_TPM_PBS",
  output_png = "Volcano_BLM+antiCCR8_vs_PBS_by_padj.png",
  plot_title = "Volcano Plot: BLM+antiCCR8 vs PBS (padj)"
)

volcano_plot_padj(
  input_csv = "DESeq2_BLM+IgG_vs_PBS.csv",
  tpm_col1 = "mean_TPM_BLM + IgG",
  tpm_col2 = "mean_TPM_PBS",
  output_png = "Volcano_BLM+IgG_vs_PBS_by_padj.png",
  plot_title = "Volcano Plot: BLM+IgG vs PBS (padj)"
)


# GO Analysis (Figure 5D) 
perform_analysis <- function(input_csv, comparison_label) {
  res_df <- read.csv(input_csv, check.names = FALSE)
  significant_genes <- res_df %>% filter(padj < 0.05, abs(log2FoldChange) > 1)
  deg_genes <- significant_genes$symbol
  
  go_bp_results <- enrichGO(gene = deg_genes, OrgDb = org.Mm.eg.db, keyType = "SYMBOL", ont = "BP", pvalueCutoff = 0.05)
  go_mf_results <- enrichGO(gene = deg_genes, OrgDb = org.Mm.eg.db, keyType = "SYMBOL", ont = "MF", pvalueCutoff = 0.05)
  
  write.xlsx(as.data.frame(go_bp_results@result), file = paste0(comparison_label, "_GO_BP.xlsx"), rowNames = TRUE)
  write.xlsx(as.data.frame(go_mf_results@result), file = paste0(comparison_label, "_GO_MF.xlsx"), rowNames = TRUE)
  
  p_bp <- dotplot(go_bp_results, showCategory = 10, title = paste0(comparison_label, " GO Biological Process")) +
    theme(axis.text = element_text(size=14), axis.title = element_text(size=14))
  ggsave(paste0(comparison_label, "_GO_BP.png"), plot = p_bp, width = 7, height = 5, dpi = 300)
  
  p_mf <- dotplot(go_mf_results, showCategory = 10, title = paste0(comparison_label, " GO Molecular Function")) +
    theme(axis.text = element_text(size=12), axis.title = element_text(size=14))
  ggsave(paste0(comparison_label, "_GO_MF.png"), plot = p_mf, width = 7, height = 5, dpi = 300)
}

perform_analysis("Filtered_DESeq2_BLM+antiCCR8_vs_PBS.csv", "Filtered_BLM+antiCCR8_vs_PBS")
perform_analysis("Filtered_DESeq2_BLM+IgG_vs_PBS.csv", "Filtered_BLM+IgG_vs_PBS")


# GSEA Analysis (Figure 5E)

perform_gsea_analysis <- function(input_csv, comparison_label) {
  
  res_df <- read.csv(input_csv, check.names = FALSE)
  geneList <- res_df$log2FoldChange
  names(geneList) <- res_df$symbol
  geneList <- sort(geneList, decreasing = TRUE)
  
  # Hallmark GSEA
  hallmark_gene_sets <- msigdbr(species = "Mus musculus", category = "H") %>%
    dplyr::select(gs_name, gene_symbol)
  
  gsea_hallmark <- GSEA(geneList,
                        TERM2GENE = hallmark_gene_sets,
                        pvalueCutoff = 0.1,
                        verbose = FALSE)
  
  write.xlsx(as.data.frame(gsea_hallmark@result),
             file = paste0(comparison_label, "_GSEA_Hallmark.xlsx"),
             rowNames = TRUE)
  
  p_hallmark <- ggplot(as.data.frame(gsea_hallmark@result),
                       aes(x = reorder(ID, NES), y = NES, fill = NES)) +
    geom_col() +
    coord_flip() +
    theme_bw(base_size = 16) +
    labs(title = paste0(comparison_label, " Hallmark GSEA Pathways"),
         x = "Pathway", y = "Normalized Enrichment Score (NES)") +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
    scale_x_discrete(labels = function(x) str_wrap(x, width = 60))
  
  ggsave(paste0(comparison_label, "_GSEA_Hallmark.png"),
         plot = p_hallmark, width = 12, height = 9, dpi = 300)
  
  # KEGG-based GSEA
  gene_mapping <- bitr(names(geneList),
                       fromType = "SYMBOL", toType = "ENTREZID",
                       OrgDb = org.Mm.eg.db)
  
  geneList_kegg <- geneList[gene_mapping$SYMBOL]
  names(geneList_kegg) <- gene_mapping$ENTREZID
  geneList_kegg <- sort(geneList_kegg, decreasing = TRUE)
  
  gsea_kegg <- gseKEGG(geneList = geneList_kegg,
                       organism = 'mmu',
                       pvalueCutoff = 0.05,
                       verbose = FALSE)
  
  write.xlsx(as.data.frame(gsea_kegg@result),
             file = paste0(comparison_label, "_GSEA_KEGG.xlsx"),
             rowNames = TRUE)
  
  p_kegg <- ggplot(as.data.frame(gsea_kegg@result),
                   aes(x = reorder(Description, NES), y = NES, fill = NES)) +
    geom_col() +
    coord_flip() +
    theme_bw(base_size = 16) +
    labs(title = paste0(comparison_label, " KEGG-based GSEA Pathways"),
         x = "Pathway", y = "Normalized Enrichment Score (NES)") +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
    scale_x_discrete(labels = function(x) str_wrap(x, width = 60))
  
  ggsave(paste0(comparison_label, "_GSEA_KEGG.png"),
         plot = p_kegg, width = 12, height = 9, dpi = 300)
}

perform_gsea_analysis("Filtered_DESeq2_BLM+antiCCR8_vs_PBS.csv", "BLM+antiCCR8_vs_PBS")
perform_gsea_analysis("Filtered_DESeq2_BLM+IgG_vs_PBS.csv", "BLM+IgG_vs_PBS")

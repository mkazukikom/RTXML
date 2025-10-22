# 必要なパッケージの読み込み
library(WGCNA)
library(dplyr)
library(clusterProfiler)
library(org.Hs.eg.db)
options(stringsAsFactors = FALSE)

# データの読み込み
data <- read.csv("rtx_data_matched.csv")

# A4GALT以降の列を選択
genesData <- data[, which(names(data) == "A4GALT"):ncol(data)]

# 各列の最大値が1.5625未満かどうかをチェックします
maxCheck <- apply(genesData, 2, function(x) max(x, na.rm = TRUE) < 1.5625)

# 条件を満たさない列を除外します
genesData <- genesData[, !maxCheck]

# 0から1の範囲で正規化
normalize_0_to_1 <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
n_genesData <- as.data.frame(lapply(genesData, normalize_0_to_1))

# データの前処理: 欠損値の扱いやサンプル/遺伝子のフィルタリングなど
#gene_mad = apply(t(n_genesData), 1, mad)
#hist(gene_mad)
#gene_mad_rank = rank(-gene_mad, ties.method = "first")
#keep = gene_mad_rank < 6000
#hist(gene_mad[keep])
#n_genesData <- n_genesData[,keep]

# 軟閾値パワーの選択
powers <- c(seq(1, 10, by=1), seq(12, 20, by=2))
sft <- pickSoftThreshold(n_genesData, powerVector = powers, verbose = 5)
sizeGrWindow(9, 5)
par(mfrow = c(1,2))
cex1 = 0.9
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     xlab="Soft Threshold (power)", ylab="Scale Free Topology Model Fit, signed R^2",
     type="n", main = paste("Scale independence"))
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     labels=powers,cex=cex1,col="red")

# ネットワークの構築とモジュールの検出
softPower <- 6 # ここでは例として6を使用していますが、上のプロットから適切な値を選択してください。
adjacency <- adjacency(n_genesData, power = softPower)
k=as.vector(apply(adjacency, 2, sum, na.rm=T))
hist(k)
scaleFreePlot(k, main="Check scale free topology\n")
TOM <- TOMsimilarity(adjacency)
dissTOM = 1-TOM
geneTree = hclust(as.dist(dissTOM), method = "average")
minModuleSize = 30
dynamicMods = cutreeDynamic(dendro = geneTree, distM = dissTOM,
                            deepSplit = 2, pamRespectsDendro = FALSE,
                            minClusterSize = minModuleSize)
moduleColors = labels2colors(dynamicMods)
table(moduleColors)

plotDendroAndColors(dendro = geneTree, 
                    colors = moduleColors, 
                    groupLabels = "Dynamic Tree Cut",
                    dendroLabels = FALSE, hang = 0.03,
                    addGuide = TRUE, guideHang = 0.05,
                    main = "Gene dendrogram and module colors")

MEList <- moduleEigengenes(n_genesData, colors = moduleColors)
MEs <- MEList$eigengenes
MEDiss <- 1-cor(MEs,use = 'pairwise.complete.obs')
METree <- hclust(as.dist(MEDiss), method = "average")
plot(METree, main = "Clustering of module eigengenes",xlab = "", sub = "")
MEDissThres <- 0.15
abline(h=MEDissThres, col = "red")

TOMplot(TOM, geneTree, moduleColors, main = "Network heatmap plot")

# rtx_traitデータの読み込み
traits <- read.csv("rtx_trait_matched.csv")

# "Age"列以降のtraitを抽出
traits <- traits[, 4:41]

# モジュールの代表値（たとえば、各モジュールの主成分）を計算
MEs <- moduleEigengenes(n_genesData, colors = moduleColors)$eigengenes

# traitとモジュールの固有遺伝子との相関を計算
corAndPvalue <- corAndPvalue(MEs, traits)

# 相関係数とP値のマトリックスを抽出
corMatrix <- corAndPvalue$cor
pvalMatrix <- corAndPvalue$p

# 相関係数のヒートマップ表示
#pheatmap(corMatrix,
#         display_numbers = matrix(sprintf("%.2f\np=%.3f", corMatrix, pvalMatrix), ncol=ncol(corMatrix)),
#         color = colorRampPalette(c("blue", "white", "red"))(100),
#         annotation_legend = TRUE)

library(circlize)
library(ComplexHeatmap)

heatmap_data <- corMatrix

# モジュールカラーの設定（仮データ）
color_names <- gsub("ME", "", row.names(heatmap_data))
# クラスタリング後の行の順序を取得
heatmap_object = draw(heatmap_object); row_order(heatmap_object)
# color_namesの順序をクラスタリング結果に合わせて調整
adjusted_color_names <- color_names[row_order]

# P値に基づくアノテーション（仮データを使用しています）
pval_annotation <- matrix("", nrow = nrow(pvalMatrix), ncol = ncol(pvalMatrix))
pval_annotation[pvalMatrix < 0.001] <- "***"
pval_annotation[pvalMatrix < 0.01 & pvalMatrix >= 0.001] <- "**"
pval_annotation[pvalMatrix < 0.05 & pvalMatrix >= 0.01] <- "*"

# 左側アノテーションの作成
ha_left <- rowAnnotation(row_color = anno_block(gp = gpar(fill = adjusted_color_names)))

# ヒートマップの作成
Heatmap(heatmap_data, name = "Correlation",
#                left_annotation = ha_left,
        cell_fun = function(j, i, x, y, width, height, fill) {
          grid.text(pval_annotation[i, j], x, y, gp = gpar(fontsize = 10))
        },
        col = colorRampPalette(c("blue", "white", "red"))(100))

# 'TOP1'遺伝子のインデックスを取得します
geneOfInterest <- "TOP1"
geneIndex <- which(genes == geneOfInterest)
print(moduleColors[geneIndex])

geneOfInterest <- "CENPB"
geneIndex <- which(genes == geneOfInterest)
print(moduleColors[geneIndex])

geneOfInterest <- "POLR3A_D"
geneIndex <- which(genes == geneOfInterest)
print(moduleColors[geneIndex])

geneOfInterest <- "CCR8"
geneIndex <- which(genes == geneOfInterest)
print(moduleColors[geneIndex])

geneOfInterest <- "NPFFR2"
geneIndex <- which(genes == geneOfInterest)
print(moduleColors[geneIndex])

geneOfInterest <- "P2RY8"
geneIndex <- which(genes == geneOfInterest)
print(moduleColors[geneIndex])

geneOfInterest <- "MC1R"
geneIndex <- which(genes == geneOfInterest)
print(moduleColors[geneIndex])

geneOfInterest <- "HTR1B"
geneIndex <- which(genes == geneOfInterest)
print(moduleColors[geneIndex])

geneOfInterest <- "FPR1"
geneIndex <- which(genes == geneOfInterest)
print(moduleColors[geneIndex])

# 特定のモジュールに属する遺伝子のインデックスを特定
GeneIndices <- which(moduleColors == "salmon")

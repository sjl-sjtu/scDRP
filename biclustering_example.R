library(tidyverse)
library(biclust)
library(igraph)

merge_clusters_to_fgms <- function(lists_of_gene_sets, os_thresh = 0.03, iou_thresh = 0.3) {
  # Step 1.
  for (i in seq_along(lists_of_gene_sets)) {
    cond_name <- paste0("cond", i)
    if (is.null(names(lists_of_gene_sets[[i]]))) {
      names(lists_of_gene_sets[[i]]) <- paste0(cond_name, "_cluster", seq_along(lists_of_gene_sets[[i]]))
    } else {
      names(lists_of_gene_sets[[i]]) <- paste0(cond_name, "_", names(lists_of_gene_sets[[i]]))
    }
  }
  
  
  filter_biclusters <- function(bicluster_list) {
    filtered_biclusters <- lapply(bicluster_list, function(condition) {
      # if there is only one bicluster in a condition, do not remove
      if (length(condition) <= 1) return(condition)
      common_genes <- Reduce(intersect, lapply(condition, function(gene_set) {
        gene_set
      }))
      lapply(condition, function(gene_set) {
        setdiff(gene_set, common_genes) 
      })
    })
    return(filtered_biclusters)
  }
  lists_of_gene_sets <- filter_biclusters(lists_of_gene_sets)
  
  # helper: exclusive overlap score
  overlap_score <- function(g1, g2, all_sets) {
    inter <- intersect(g1, g2)
    others <- setdiff(all_sets, list(g1, g2))
    if (length(others) > 0) {
      inter <- setdiff(inter, unique(unlist(others)))
    }
    e <- length(inter)
    denom <- max(length(g1), length(g2))
    if (denom == 0) return(0)
    e / denom
  }
  
  # helper: IoU
  iou_score <- function(g1, g2) {
    iu <- length(intersect(g1, g2))
    un <- length(union(g1, g2))
    if (un == 0) return(0)
    iu / un
  }
  
  # Step 2.
  merge_within_condition <- function(gene_list, os_thresh) {
    if (is.null(names(gene_list))) {
      names(gene_list) <- paste0("cluster", seq_along(gene_list))
    }
    n <- length(gene_list)
    if (n == 0) return(list(groups = list(), mapping = character(0)))
    if (n == 1) {
      return(list(groups = list(G1 = list(genes = unique(gene_list[[1]]),
                                          members = names(gene_list))),
                  mapping = setNames("G1", names(gene_list))))
    }
    
    # 1. 计算所有 pair overlap score
    all_genesets <- lapply(gene_list, unique)
    pair_idx <- combn(seq_len(n), 2, simplify = FALSE)
    edges <- lapply(pair_idx, function(idx) {
      i <- idx[1]; j <- idx[2]
      s <- overlap_score(all_genesets[[i]], all_genesets[[j]], all_genesets)
      if (s > os_thresh) return(c(i, j)) else return(NULL)
    })
    edges <- do.call(rbind, edges[!sapply(edges, is.null)])
    
    if (is.null(edges)) {
      groups <- lapply(names(gene_list), function(nm) {
        list(genes = unique(gene_list[[nm]]), members = nm)
      })
      names(groups) <- paste0("G", seq_along(groups))
      mapping <- unlist(lapply(seq_along(groups), function(i) {
        setNames(rep(names(groups)[i], length(groups[[i]]$members)),
                 groups[[i]]$members)
      }))
      return(list(groups = groups, mapping = mapping))
    }
    
    g <- igraph::graph_from_edgelist(edges, directed = FALSE)
    comps <- igraph::components(g)$membership
    
    groups <- list()
    for (comp_id in unique(comps)) {
      idxs <- which(comps == comp_id)
      members <- names(gene_list)[idxs]
      merged_genes <- unique(unlist(all_genesets[idxs]))
      groups[[paste0("G", comp_id)]] <- list(
        genes = merged_genes,
        members = members
      )
    }
    
    remaining_clusters <- setdiff(names(gene_list), unlist(lapply(groups, function(g) g$members)))
    for (remaining in remaining_clusters) {
      groups[[paste0("G", length(groups) + 1)]] <- list(
        genes = unique(gene_list[[remaining]]),
        members = remaining
      )
    }
    
    mapping <- unlist(lapply(names(groups), function(gid) {
      setNames(rep(gid, length(groups[[gid]]$members)), groups[[gid]]$members)
    }))
    
    return(list(groups = groups, mapping = mapping))
  }
  
  
  # Step 3.
  cond_names <- names(lists_of_gene_sets)
  if (is.null(cond_names)) cond_names <- paste0("Cond", seq_along(lists_of_gene_sets))
  within_results <- vector("list", length(lists_of_gene_sets))
  names(within_results) <- cond_names
  for (i in seq_along(lists_of_gene_sets)) {
    within_results[[i]] <- merge_within_condition(lists_of_gene_sets[[i]], os_thresh)
  }
  
  # Step 4.
  group_entries <- list()
  for (cond in cond_names) {
    res <- within_results[[cond]]
    groups <- res$groups
    for (gname in names(groups)) {
      global_id <- paste0(cond, "__", gname)
      group_entries[[global_id]] <- list(
        global_id = global_id,
        cond = cond,
        genes = unique(groups[[gname]]$genes),
        members = groups[[gname]]$members
      )
    }
  }
  
  all_group_ids <- names(group_entries)
  n_groups <- length(all_group_ids)
  if (n_groups == 0) {
    return(list(condition_fgm_tables = list(),
                fgm_genes = list(),
                cluster_to_fgm_df = data.frame()))
  }
  
  pair_idx <- combn(seq_len(n_groups), 2, simplify = FALSE)
  edges <- lapply(pair_idx, function(idx) {
    id_i <- all_group_ids[idx[1]]
    id_j <- all_group_ids[idx[2]]
    cond_i <- group_entries[[id_i]]$cond
    cond_j <- group_entries[[id_j]]$cond
    if (cond_i == cond_j) return(NULL)
    s <- iou_score(group_entries[[id_i]]$genes, group_entries[[id_j]]$genes)
    if (s > iou_thresh) return(c(id_i, id_j)) else return(NULL)
  })
  edges <- do.call(rbind, edges[!sapply(edges, is.null)])
  
  if (is.null(edges) || nrow(edges) == 0) {
    comps <- seq_len(n_groups)
    names(comps) <- all_group_ids
  } else {
    g <- igraph::graph_from_edgelist(edges, directed = FALSE)
    missing_nodes <- setdiff(all_group_ids, V(g)$name)
    if (length(missing_nodes) > 0) {
      g <- igraph::add_vertices(g, nv = length(missing_nodes), name = missing_nodes)
    }
    comps <- igraph::components(g)$membership
  }
  
  unique_comps <- unique(comps)
  fgm_ids <- paste0("FGM", seq_along(unique_comps))
  comp_to_fgm <- setNames(fgm_ids, unique_comps)
  group_to_fgm <- setNames(comp_to_fgm[as.character(comps)], names(comps))

  fgm_genes <- list()
  for (fid in unique(group_to_fgm)) {
    members_groups <- names(group_to_fgm)[group_to_fgm == fid]
    genes_union <- unique(unlist(lapply(members_groups, function(gid) group_entries[[gid]]$genes)))
    fgm_genes[[fid]] <- genes_union
  }
  
  condition_fgm_tables <- list()
  cluster_to_fgm_rows <- list()
  for (cond in cond_names) {
    res <- within_results[[cond]]
    mapping <- res$mapping
    if (length(mapping) == 0) {
      condition_fgm_tables[[cond]] <- data.frame()
      next
    }
    rows <- lapply(names(mapping), function(orig_cl) {
      local_gid <- mapping[[orig_cl]]
      global_gid <- paste0(cond, "__", local_gid)
      fgmid <- group_to_fgm[[global_gid]]
      data.frame(condition = cond, original_cluster = orig_cl,
                 group_id = local_gid, group_global_id = global_gid,
                 FGM_id = fgmid, stringsAsFactors = FALSE)
    })
    df_cond <- do.call(rbind, rows)
    condition_fgm_tables[[cond]] <- df_cond
    cluster_to_fgm_rows[[cond]] <- df_cond
  }
  cluster_to_fgm_df <- do.call(rbind, cluster_to_fgm_rows)
  rownames(cluster_to_fgm_df) <- NULL
  
  return(list(
    condition_fgm_tables = condition_fgm_tables,
    fgm_genes = fgm_genes,
    cluster_to_fgm_df = cluster_to_fgm_df
  ))
  
}

df <- read_csv("./ite.csv")
ite <- df[,-c(ncol(df)-1,ncol(df))]|>as.matrix()

results <- list()
celltypes <- unique(df$celltype)
perturbations <- unique(df$perturbation)
for(perturbation in perturbations){
  for(celltype in celltypes){
    ite1 <- ite[(df$celltype==celltype) & (df$perturbation==perturbation),]
    res1 <- biclust(ite1, method = BCPlaid(),fit.model = ~m + a + b, row.release = 0.7, 
                    col.release = 0.7)
    results[celltype] <- res1
  }
}


lists_of_gene_sets <- list()
for(name in names(results)){
  genes <- list()
  res <- results[[name]]
  for(i in 1:res@Number){
    genes[[i]] <- colnames(ite)[res@NumberxCol[i,]]
  }
  lists_of_gene_sets[[name]] <- genes
}

res <- merge_clusters_to_fgms(lists_of_gene_sets, os_thresh = 0.03, iou_thresh = 0.3)

# assign cells to FGMs
n_fgm <- length(res$fgm_genes)
# judge each cell belongs to which FGM

outs_list <- list()
for(i in seq_along(results)){
  res1 <- results[[i]]
  name <- names(results)[i]
  if(name=="all"){
    next
  }
  cell_df <- as.data.frame(res1@RowxNumber)
  colnames(cell_df) <- paste0("cond",i,"_cluster", seq_len(res1@Number))
  cell_df <- cell_df %>% mutate(Cell = df[df$celltype==name,]$cell_name)
  cell_long <- cell_df %>%
    pivot_longer(-Cell, names_to = "original_cluster", values_to = "Membership") %>%
    filter(Membership == 1)
  outs <- cell_long %>%
    left_join(res$cluster_to_fgm_df, by = "original_cluster") %>%
    dplyr::select(Cell, FGM_id) %>%
    distinct()
  # mutate(Present = 1) %>%
  # pivot_wider(names_from = FGM_id, values_from = Present, values_fill = list(Present = 0))
  outs_list[[name]] <- outs
}
df_ous <- do.call(rbind, outs_list)
df_ous <- df_ous %>% left_join(df[,c('cell_name','celltype')], by = c("Cell" = "cell_name"))

composition_prop <- df_ous %>%
  group_by(celltype, FGM_id) %>%
  summarise(Count = dplyr::n(), .groups = "drop") %>%  
  group_by(celltype) %>%
  mutate(Percentage = Count / sum(Count) * 100) %>%
  ungroup()
composition_prop

library(clusterProfiler)
library(org.Hs.eg.db)
library(patchwork)
fgms <- names(res$fgm_genes)
for(fgm in fgms){
  genes <- res$fgm_genes[[fgm]]
  if(length(genes)==0){
    next
  }else{
    test = bitr(genes, fromType="SYMBOL", toType="ENSEMBL", OrgDb="org.Hs.eg.db") #"org.Mm.eg.db")ENTREZID
    enrich.go <- enrichGO(gene = test$ENSEMBL, 
                          OrgDb = org.Hs.eg.db, 
                          keyType = 'ENSEMBL',
                          ont = "ALL", 
                          pAdjustMethod = "BH", 
                          pvalueCutoff = 0.05,
                          qvalueCutoff = 0.05,
                          readable = TRUE)
    print(enrich.go)
  }
}

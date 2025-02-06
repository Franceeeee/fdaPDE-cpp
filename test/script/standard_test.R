rm(list=ls())

# package installation -------------------------------------------------------------------
# install.packages(c("stats", "grDevices", "graphics", "geometry", "rgl", "Matrix", "plot3D", "plot3Drgl", "shiny"))
# install.packages("~/Desktop/fdaPDEmixed", repos = NULL, type = "source")
# if(!require(pacman)) install.packages("pacman")
# install.packages("magick")
pacman::p_load("fdaPDE" ,"plotrix", "latex2exp", "RColorBrewer", "viridis", "dplyr")
# if(!require(fdaPDEmixed)){
#   devtools::install_github(repo ="aldoclemente/fdaPDEmixed")
# }

# functions from graphic tools -----------------------------------------------------------

zoom = 0.7657689
windowRect = c(70,  106, 1920, 1117)
plot_smooth_2D <- function(FEM, coeff_lims=smooth_lim(FEM), 
                           colorscale = jet.col, ncolor = 128, alpha = 1,
                           ...){
  nodes <- FEM$FEMbasis$mesh$nodes
  triangles <- as.vector(t(FEM$FEMbasis$mesh$triangles))
  coeff = FEM$coeff 
  p = colorscale(n = ncolor, alpha = alpha)
  grDevices::palette(p)
  open3d(zoom=zoom, windowRect=windowRect)
  pop3d("lights")
  light3d(specular="black")
  
  diffrange = diff(range(coeff_lims)) 
  col = coeff[triangles,]
  col = (col - min(coeff, na.rm =T))/diffrange*(ncolor-1)+1
  if(abs(diffrange) < 1e-10) col = rep(M, times=length(col)) # costanti
  
  #z <- FEM$coeff[triangles,]
  triangles3d(nodes[triangles,1], nodes[triangles,2], 0,
              color = col,...)
  
  aspect3d(2,2,1)
  view3d(0,0)
}


extract_coeff <- function(FEMObject){
  elements = NULL
  mesh = FEMObject$FEMbasis$mesh
  if( is(mesh, "mesh.2D") | is(mesh, "mesh.2.5D")){
    elements = mesh$triangles
  }else if( is(mesh, "mesh.1.5D")){
    elements = mesh$edges
  }else{
    elements = mesh$tetrahedrons # tetrahedrons
  }
  coeff <- apply(elements, MARGIN=1, FUN = function(row){
    mean(FEMObject$coeff[row,])
  })
  return(coeff)
}

smooth_lim <- function(FEMObject, ...){
  coeff <- extract_coeff(FEMObject)
  lims = c(1e10, -1e10)
  lims[1] = min(coeff, lims[1], na.rm = T)
  lims[2] = max(coeff, lims[2], na.rm = T)
  
  lims[1] = min(min(FEMObject$coeff, na.rm = T), lims[1], na.rm = T)
  lims[2] = max(max(FEMObject$coeff, na.rm = T), lims[2], na.rm = T)
  
  #coeffs_args = list()
  args = list(...)
  if( length(args) > 0L){
    for(i in 1:length(args)){
      if(! is(args[[i]], "FEM") ) stop("Provides ONLY FEM objects.")
      coeff = extract_coeff(args[[i]])
      lims[1] = min(coeff, lims[1], na.rm = T)
      lims[2] = max(coeff, lims[2], na.rm = T)
      lims[1] = min(min(args[[i]]$coeff, na.rm = T), lims[1], na.rm = T)
      lims[2] = max(max(args[[i]]$coeff, na.rm = T), lims[2], na.rm = T)
    }
  }
  return(lims)
}

# ---
plot_colorbar <- function(FEMObject, coeff_lims= smooth_lim(FEMObject), 
                          colorscale = jet.col, ncolor = 128, width=3, 
                          cex.axis = 2, file = "colorbar"){
  coeff <- extract_coeff(FEMObject)
  #if(is.null(coeff_lims)) coeff_lims = c(min(coeff, na.rm = T), max(coeff, na.rm = T))
  cmin = coeff_lims[1]; cmax=coeff_lims[2]
  
  exps <- -15:15
  range_ <- round( diff(range(coeff_lims)), digits = 2)
  cmin_exp <- which(floor(log10(abs(signif(signif(cmin, digits = 2) / 10^exps, digits = 0))))==0)
  cmax_exp <- which(floor(log10(abs(signif(signif(cmax, digits = 2) / 10^exps, digits = 0))))==0)
  k <- exps[max(cmin_exp, cmax_exp)]
  
  at = seq(0, 100, length.out=5)
  labels = as.character(round(seq(cmin*10^(-k), cmax*10^(-k), length.out=5), 2))
  text_ <- ifelse(k != 0, paste0("$\\times 10^{", k,"}$"), "")
  
  diffrange = cmax - cmin 
  if(abs(diffrange) < 1e-10) ncolor = 1 # costanti
  
  labels_grad = ifelse(text_ == "", "", TeX(text_))
  
  png(paste0(file, "_horiziontal.png"), family = "serif", width = 11, height = 3, units="in", res=150)
  par(mai = c(1,0.75,0,0))
  plot(c(0, 112.5), c(0, 15), type = "n", xlab = "", ylab = "", xaxt = "n", yaxt = "n", frame.plot = F)
  gradient.rect(0, 0, 100, width, col = colorscale(ncolor), border = "black")
  axis(1, at = at, labels = labels, # at = c(0,33.33,66.66,100)
       cex.axis = cex.axis, lwd.ticks = 0, lwd = 0) # lwd.ticks = 2, 
  text(107,2, labels_grad, cex = 2)
  dev.off()
  
  png(paste0(file, "_vertical.png"), family = "serif", width = 3, height = 11,  units="in", res=150)
  par(mai = c(1,0.75,0,0))
  plot(c(0, 15), c(0, 112.5), type = "n", xlab = "", ylab = "", xaxt = "n", yaxt = "n", frame.plot = F)
  gradient.rect(0, 0, width, 100, col = colorscale(ncolor), border = "black", gradient = "y")
  axis(4, at = at, labels = labels, # at = c(0,33.33,66.66,100)
       cex.axis = cex.axis, lwd.ticks = 0, lwd = 0,  line=-11.5+width) # lwd.ticks = 2, 
  text(2.5, 107, labels_grad, cex = 2)
  dev.off()
}

# ---
# n_obs  stringa -> nome della colonna, factor, di data che contiene numero osservazioni / numero nodi della mesh
# method stringa -> nome della colonna, factor, di data che contiene i metodi
plot_boxplot = function(data, n_obs, method,
                        filename="boxplot.pdf"){
  n_obs_ = as.numeric(levels(data[[n_obs]]))
  methods_ = levels(data[[method]])
  at_ <- c()
  for(i in 1:length(n_obs_)){
    at_ <-  c(at_, ((i-1)*(1+length(methods_)) + (1:length(methods_))))
  }
  
  fill_col = viridis::viridis((length(levels(data[[method]]))+1), begin=0.25, end=0.95)
  fill_col = fill_col[1:length(methods_)]
  facs = names(Filter(is.factor, data))
  which( ! names(data) %in% facs )
  doplot = names(data)[which( ! names(data) %in% facs )]
  
  if(length(methods_)%%2 != 0){
    at_label = seq(ceiling(length(methods)/2), at_[length(at_)], by=(length(methods_) + 1))
  }else{
    at_label = seq(length(methods_)/2, at_[length(at_)], by=(length(methods_) + 1))
  }
  
  pdf(filename, family = "serif", width = 7, height = 7)
  for(i in doplot){
    boxplot(data[[i]] ~  data[[method]] + as.numeric(data[[n_obs]]),
            ylab="", xlab="observations", at = at_, xaxt="n",
            ylim=c(min(data[[i]]), max(data[[i]])*(1.1)),
            col=fill_col,cex.lab = 2, cex.axis = 2, cex.main = 2,
            main = i)
    axis(side = 1, at = at_label, labels = n_obs_, cex.lab = 2, cex.axis = 2)
    legend("topright",legend=methods_, fill=fill_col, horiz=T, cex=1.5, inset=0.0125, 
           bty="n")
    
  }
  
  for(k in 1:length(methods_)){
    for(i in doplot){
      tmp <- data[which(data[[method]] == methods_[k]),]
      boxplot(tmp[[i]] ~ tmp[[n_obs]],
              ylab="", xlab="observations", xaxt="n",
              col=fill_col[k],cex.lab = 2, cex.axis = 2, cex.main = 2,
              main = paste0(i," (",methods_[k],")"))
      axis(side = 1, at = 1:length(n_obs_), labels = n_obs_, cex.lab = 2, cex.axis = 2)
    }
  }
  dev.off()
}

# ---
# m  stringa -> nome della colonna, factor, di data che contiene numero di livelli
# method stringa -> nome della colonna, factor, di data che contiene i metodi
plot_boxplot_levels = function(data, m, method, xlabel_name = "levels",
                               filename="boxplot.pdf"){
  m_ = as.numeric(levels(data[[m]]))
  methods_ = levels(data[[method]])
  at_ <- c()
  for(i in 1:length(m_)){
    at_ <-  c(at_, ((i-1)*(1+length(methods_)) + (1:length(methods_))))
  }
  
  fill_col = viridis::viridis((length(levels(data[[method]]))+1), begin=0.25, end=0.95)
  fill_col = fill_col[1:length(methods_)]
  facs = names(Filter(is.factor, data))
  which( ! names(data) %in% facs )
  doplot = names(data)[which( ! names(data) %in% facs )]
  
  if(length(methods_)%%2 != 0){
    at_label = seq(ceiling(length(methods)/2), at_[length(at_)], by=(length(methods_) + 1))
  }else{
    at_label = seq(length(methods_)/2, at_[length(at_)], by=(length(methods_) + 1))
  }
  
  pdf(filename, family = "serif", width = 7, height = 7)
  for(i in doplot){
    boxplot(data[[i]] ~  data[[method]] + as.numeric(data[[m]]),
            ylab="", xlab=xlabel_name, at = at_, xaxt="n",
            ylim=c(min(data[[i]]), max(data[[i]])*(1.1)),
            col=fill_col,cex.lab = 2, cex.axis = 2, cex.main = 2,
            main = i)
    axis(side = 1, at = at_label, labels = m_, cex.lab = 2, cex.axis = 2)
    legend("topright",legend=methods_, fill=fill_col, cex=1.5, inset=0.0125, 
           bty="n")
    
  }
  
  for(k in 1:length(methods_)){
    for(i in doplot){
      tmp <- data[which(data[[method]] == methods_[k]),]
      boxplot(tmp[[i]] ~ tmp[[m]],
              ylab="", xlab="observations", xaxt="n",
              col=fill_col[k],cex.lab = 2, cex.axis = 2, cex.main = 2,
              main = paste0(i," (",methods_[k],")"))
      axis(side = 1, at = 1:length(m_), labels = m_, cex.lab = 2, cex.axis = 2)
    }
  }
  dev.off()
}

plot_mesh <- function(data_dir, mesh_plot_name, n_obs, sim, solution_policy){
  
  mesh_dir = "//wsl.localhost/Ubuntu/root/fdaPDE-cpp/test/data/mesh/"
  nodes = read.csv(paste0(mesh_dir, mesh_id, "/points.csv"))[,2:3]
  triangles = read.csv(paste0(mesh_dir, mesh_id, "/elements.csv"))[,2:4]
  
  mesh=create.mesh.2D(nodes=nodes, triangles = triangles)
  FEMbasis <- create.FEM.basis(mesh)
  
  output_mono_dir = paste0(data_dir,"output/",n_obs, sim,solution_policy[1],"/")
  output_mono_cpp = list(coeff = as.matrix(read.table(paste0(output_mono_dir, "estimate_f.txt"), header = F)),
                         beta = as.matrix(read.table(paste0(output_mono_dir,"beta.txt"), header = F)),
                         alpha_i = as.matrix(read.table(paste0(output_mono_dir,"alpha.txt"), header = F)),
                         f_0 = as.matrix(read.table(paste0(output_mono_dir,"estimate_f_0.txt"), header = F)),
                         f_1 = as.matrix(read.table(paste0(output_mono_dir,"estimate_f_1.txt"), header = F)),
                         f_2 = as.matrix(read.table(paste0(output_mono_dir,"estimate_f_2.txt"), header = F)))
  output_iter_dir = paste0(data_dir,"output/",n_obs, sim,solution_policy[2],"/")
  output_iter_cpp = list(coeff = as.matrix(read.table(paste0(output_iter_dir, "estimate_f.txt"), header = F)),
                         beta = as.matrix(read.table(paste0(output_iter_dir,"beta.txt"), header = F)),
                         alpha_i = as.matrix(read.table(paste0(output_iter_dir,"alpha.txt"), header = F)),
                         f_0 = as.matrix(read.table(paste0(output_iter_dir,"estimate_f_0.txt"), header = F)),
                         f_1 = as.matrix(read.table(paste0(output_iter_dir,"estimate_f_1.txt"), header = F)),
                         f_2 = as.matrix(read.table(paste0(output_iter_dir,"estimate_f_2.txt"), header = F)))
  
  imgdir = "imgs/"
  if(!dir.exists(imgdir)) dir.create(imgdir)
  
  imgdir = paste0(imgdir, test_id, "/")
  if(!dir.exists(imgdir)) dir.create(imgdir)
  
  estimates_dir = paste0(imgdir, sim)
  if(!dir.exists(estimates_dir)) dir.create(estimates_dir)
  
  f_0_cpp_mono = FEM(output_mono_cpp$f_0, FEMbasis)
  f_0_cpp_iter = FEM(output_mono_cpp$f_0, FEMbasis)
  f_0 = FEM(as.matrix(read.table(paste0(data_dir,"input/f_0.txt"), header = F)), FEMbasis)
  
  f_1_cpp_mono = FEM(output_mono_cpp$f_1, FEMbasis)
  f_1_cpp_iter = FEM(output_mono_cpp$f_1, FEMbasis)
  f_1 = FEM(as.matrix(read.table(paste0(data_dir,"input/f_1.txt"), header = F)), FEMbasis)
  
  f_2_cpp_mono = FEM(output_mono_cpp$f_2, FEMbasis)
  f_2_cpp_iter = FEM(output_mono_cpp$f_2, FEMbasis)
  f_2 = FEM(as.matrix(read.table(paste0(data_dir,"input/f_2.txt"), header = F)), FEMbasis)
  
  coeff_lims_0 = smooth_lim(f_0, f_0_cpp_mono, f_0_cpp_mono)
  coeff_lims_1 = smooth_lim(f_1, f_1_cpp_mono, f_1_cpp_mono)
  coeff_lims_2 = smooth_lim(f_2, f_2_cpp_mono, f_2_cpp_mono)
  
  {
    smooth_list = list(f_0 = f_0, f_0_cpp_iter = f_0_cpp_iter, f_0_cpp_mono = f_0_cpp_mono)
    names(smooth_list)
    for(i in 1:length(smooth_list)){
      plot_smooth_2D(smooth_list[[i]], coeff_lims = coeff_lims_0, colorscale = viridis)
      snapshot3d(filename = paste0(estimates_dir, names(smooth_list)[i],".png"),
                 fmt = "png", width = 800, height = 750, webshot = rgl.useNULL())
      close3d()  
    }
    
    plot_colorbar(f_0, coeff_lims = coeff_lims_0, colorscale = viridis,
                  file = paste0(estimates_dir, "colorbar_f_0"))
  }
  
  {
    smooth_list = list(f_1 = f_1, f_1_cpp_iter = f_1_cpp_iter, f_1_cpp_mono = f_1_cpp_mono)
    names(smooth_list)
    for(i in 1:length(smooth_list)){
      plot_smooth_2D(smooth_list[[i]], coeff_lims = coeff_lims_1, colorscale = viridis)
      snapshot3d(filename = paste0(estimates_dir, names(smooth_list)[i],".png"),
                 fmt = "png", width = 800, height = 750, webshot = rgl.useNULL())
      close3d()  
    }
    
    plot_colorbar(f_1, coeff_lims = coeff_lims_1, colorscale = viridis,
                  file = paste0(estimates_dir, "colorbar_f_1"))
    
  }
  
  {
    smooth_list = list(f_2 = f_2, f_2_cpp_iter = f_2_cpp_iter, f_2_cpp_mono = f_2_cpp_mono)
    names(smooth_list)
    for(i in 1:length(smooth_list)){
      plot_smooth_2D(smooth_list[[i]], coeff_lims = coeff_lims_2, colorscale = viridis)
      snapshot3d(filename = paste0(estimates_dir, names(smooth_list)[i],".png"),
                 fmt = "png", width = 800, height = 750, webshot = rgl.useNULL())
      close3d()  
    }
    
    plot_colorbar(f_2, coeff_lims = coeff_lims_2, colorscale = viridis,
                  file = paste0(estimates_dir, "colorbar_f_2"))
    
  }
  
  
  {
    
    imgdir = "imgs/"
    if(!dir.exists(imgdir)) dir.create(imgdir)
    
    imgdir = paste0(imgdir, test_id, "/")
    if(!dir.exists(imgdir)) dir.create(imgdir)
    
    estimates_dir = paste0(imgdir, sim)
    if(!dir.exists(estimates_dir)) dir.create(estimates_dir)
    
    r1_c1 <- image_read(paste0(estimates_dir, "f_0.png"))
    r1_c2 <- image_read(paste0(estimates_dir, "f_1.png"))
    r1_c3 <- image_read(paste0(estimates_dir, "f_2.png"))
    row1  <- image_append(c(r1_c1, r1_c2, r1_c3))
    r2_c1 <- image_read(paste0(estimates_dir,"f_0_cpp_mono.png"))
    r2_c2 <- image_read(paste0(estimates_dir,"f_1_cpp_mono.png"))
    r2_c3 <- image_read(paste0(estimates_dir,"f_2_cpp_mono.png"))
    row2  <- image_append(c(r2_c1, r2_c2, r2_c3))
    r3_c1 <- image_read(paste0(estimates_dir,"f_0_cpp_iter.png"))
    r3_c2 <- image_read(paste0(estimates_dir,"f_1_cpp_iter.png"))
    r3_c3 <- image_read(paste0(estimates_dir,"f_2_cpp_iter.png"))
    row3  <- image_append(c(r3_c1, r3_c2, r3_c3))
    
    r4_c1 <- image_resize(image_read(paste0(estimates_dir,"colorbar_f_0_horiziontal.png")), paste0(image_info(r1_c1)$width, "x"))
    r4_c2 <- image_resize(image_read(paste0(estimates_dir,"colorbar_f_1_horiziontal.png")), paste0(image_info(r1_c2)$width, "x"))
    r4_c3 <- image_resize(image_read(paste0(estimates_dir,"colorbar_f_2_horiziontal.png")), paste0(image_info(r1_c3)$width, "x"))
    row4  <- image_append(c(r4_c1, r4_c2, r4_c3))
    
    grid <- image_append(c(row1, row2, row3, row4), stack = TRUE)
    
    header1 <- image_annotate(
      image_blank(image_info(r1_c1)$width, 50, color="white"),
      text    = "f_0",
      size    = 50,
      gravity = "center"
    )
    header2 <- image_annotate(
      image_blank(image_info(r1_c2)$width, 50, color="white"),
      text    = "f_1",
      size    = 50,
      gravity = "center"
    )
    header3 <- image_annotate(
      image_blank(image_info(r1_c3)$width, 50, color="white"),
      text    = "f_2",
      size    = 50,
      gravity = "center"
    )
    
    header_row <- image_append(c(header1, header2, header3))
    grid_with_header <- image_append(c(header_row, grid), stack = TRUE)
    
    label1 <- image_annotate(
      image_blank(250, image_info(row1)$height, color="white"),
      text    = "f_true",
      size    = 50,
      gravity = "center"
    )
    label2 <- image_annotate(
      image_blank(250, image_info(row2)$height, color="white"),
      text    = "monolithic\n f_estimate",
      size    = 50,
      gravity = "center"
    )
    label3 <- image_annotate(
      image_blank(250, image_info(row3)$height, color="white"),
      text    = "iterative\n f_estimate",
      size    = 50,
      gravity = "center"
    )
    label4 <- image_annotate(
      image_blank(120, image_info(row4)$height, color="white"),
      text    = "",
      size    = 30,
      gravity = "center"
    )
    
    label_col <- image_append(c(label1, label2, label3, label4), stack = TRUE)
    final_labeled <- image_append(c(label_col, grid_with_header), stack = FALSE)
    image_write(final_labeled, paste0(imgdir,mesh_plot_name))
  }
}


# import libraries -----------------------------------------------------------------------

library(fdaPDEmixed)
library(magick)

setwd("C:/Users/Ortolani Giulia/Documents/Local/graphic-tools/")


# TEST -----------------------------------------------------------------------------------

draw_boxplots_na_fixed <- function(test_name, mesh_id, test_id, na_perc){
  data_dir = paste0("//wsl.localhost/Ubuntu/root/fdaPDE-cpp/test/data/models/mixed_srpde/", mesh_id,"/", test_id, "/")
  
  imgdir = "imgs/"
  if(!dir.exists(imgdir)) dir.create(imgdir)
  
  imgdir = paste0(imgdir, test_id, "/")
  if(!dir.exists(imgdir)) dir.create(imgdir)
  
  mono = read.table(paste0(data_dir,"output/monolithic.txt"), header = T) 
  rich = read.table(paste0(data_dir,"output/richardson.txt"), header = T)
  gmres = read.table(paste0(data_dir,"output/richardson_gmres.txt"), header = T)
  
  mono$solution_policy = rep("monolithic", times = nrow(mono))
  rich$solution_policy = rep("richardson", times = nrow(rich))
  gmres$solution_policy = rep("richardson_gmres", times = nrow(gmres))
  
  results = rbind(mono, rich, gmres)
  # results$mesh = rep(mesh_id, times = nrow(results))
  # results$solution_policy = as.factor(results$solution_policy)
  results = results[results$na_perc == na_perc, ]
  # results = results[results$rmse_f < 10, ]
  # results = results[results$rmse_f_3 < 10, ]
  # results = results[results$n_obs == 2000,]
  
  results$solution_policy = as.factor(results$solution_policy)
  results$n_obs = as.factor(results$n_obs)
  
  # plots 
  filename = paste0(test_name,".pdf")
  plot_boxplot_levels(results, m="n_obs", method="solution_policy", xlabel_name="Number of observations",
                      filename = paste0(paste0(imgdir,filename)))
}

draw_boxplots_obs_fixed <- function(test_name, mesh_id, test_id, n_obs){
  data_dir = paste0("//wsl.localhost/Ubuntu/root/fdaPDE-cpp/test/data/models/mixed_srpde/", mesh_id,"/", test_id, "/")
  
  imgdir = "imgs/"
  if(!dir.exists(imgdir)) dir.create(imgdir)
  
  imgdir = paste0(imgdir, test_id, "/")
  if(!dir.exists(imgdir)) dir.create(imgdir)
  
  mono = read.table(paste0(data_dir,"output/monolithic.txt"), header = T) 
  rich = read.table(paste0(data_dir,"output/richardson.txt"), header = T)
  gmres = read.table(paste0(data_dir,"output/richardson_gmres.txt"), header = T)
  
  mono$solution_policy = rep("monolithic", times = nrow(mono))
  rich$solution_policy = rep("richardson", times = nrow(rich))
  gmres$solution_policy = rep("richardson_gmres", times = nrow(gmres))
  
  results = rbind(mono, rich, gmres)
  # results$mesh = rep(mesh_id, times = nrow(results))
  # results$solution_policy = as.factor(results$solution_policy)
  # results = results[results$na_perc == na_perc, ]
  # results = results[results$rmse_f < 10, ]
  # results = results[results$rmse_f_3 < 10, ]
  results = results[results$n_obs == n_obs,]
  
  results$solution_policy = as.factor(results$solution_policy)
  results$na_perc = as.factor(results$na_perc)
  
  # plots 
  filename = paste0(test_name,".pdf")
  plot_boxplot_levels(results, m="na_perc", method="solution_policy", xlabel_name="NA percentage",
                      filename = paste0(paste0(imgdir,filename)))
}


##### Same locations for each level 

## 0% NA, mesh: unit_square_coarse, obs:[500, 1000, 2000, 4000, 8000]
test_name = "boxplot_na_00_unit_square_coarse"
mesh_id = "unit_square_coarse"
test_id = "same_locations_diff_NA"
draw_boxplots_na_fixed(test_name, mesh_id, test_id, na_perc = 0.0)
mesh_plot_name = "same-mesh-comparison-na-0.png"
data_dir = paste0("//wsl.localhost/Ubuntu/root/fdaPDE-cpp/test/data/models/mixed_srpde/", mesh_id,"/", test_id, "/")
n_obs = "2000/"
sim = "40/"
solution_policy = c("monolithic", "richardson")
plot_mesh(data_dir, mesh_plot_name, n_obs, sim, solution_policy)

## 10% NA, mesh: unit_square_coarse, obs:[500, 1000, 2000, 4000, 8000]
test_name = "boxplot_na_01_unit_square_coarse"
draw_boxplots_na_fixed(test_name, mesh_id, test_id, na_perc = 0.1)
mesh_plot_name = "same-mesh-comparison-na-01.png"
data_dir = paste0("//wsl.localhost/Ubuntu/root/fdaPDE-cpp/test/data/models/mixed_srpde/", mesh_id,"/", test_id, "/")
n_obs = "2000/"
sim = "40/"
solution_policy = c("monolithic", "richardson")
plot_mesh(data_dir, mesh_plot_name, n_obs, sim, solution_policy)

## 20% NA, mesh: unit_square_coarse, obs:[500, 1000, 2000, 4000, 8000]
test_name = "boxplot_na_02_unit_square_coarse"
draw_boxplots_na_fixed(test_name, mesh_id, test_id, na_perc = 0.2)
mesh_plot_name = "same-mesh-comparison-na-0.png2"
data_dir = paste0("//wsl.localhost/Ubuntu/root/fdaPDE-cpp/test/data/models/mixed_srpde/", mesh_id,"/", test_id, "/")
n_obs = "500/"
sim = "40/"
solution_policy = c("monolithic", "richardson")
plot_mesh(data_dir, mesh_plot_name, n_obs, sim, solution_policy)

## Comparison between different NA percentages, NA:[0,0.05,0.1,0.15,0.2], mesh: unit_square_coarse, obs:2000
test_name = "boxplot_obs_2000_unit_square_coarse"
mesh_id = "unit_square_coarse"
test_id = "same_locations_diff_NA"
draw_boxplots_obs_fixed(test_name, mesh_id, test_id, n_obs=2000)


##### Different locations for each level 

## 0% NA, mesh: unit_square_coarse, obs:[500, 1000, 2000, 4000, 8000]
test_name = "boxplot_na_00_unit_square_coarse"
mesh_id = "unit_square_coarse"
test_id = "diff_locations_diff_NA"
draw_boxplots_na_fixed(test_name, mesh_id, test_id, na_perc = 0.0)
mesh_plot_name = "diff-mesh-comparison-na-0.png"
data_dir = paste0("//wsl.localhost/Ubuntu/root/fdaPDE-cpp/test/data/models/mixed_srpde/", mesh_id,"/", test_id, "/")
n_obs = "2000/"
sim = "40/"
solution_policy = c("monolithic", "richardson")
plot_mesh(data_dir, mesh_plot_name, n_obs, sim, solution_policy)

## 10% NA, mesh: unit_square_coarse, obs:[500, 1000, 2000, 4000, 8000]
test_name = "boxplot_na_01_unit_square_coarse"
draw_boxplots_na_fixed(test_name, mesh_id, test_id, na_perc = 0.1)
mesh_plot_name = "diff-mesh-comparison-na-01.png"
data_dir = paste0("//wsl.localhost/Ubuntu/root/fdaPDE-cpp/test/data/models/mixed_srpde/", mesh_id,"/", test_id, "/")
n_obs = "2000/"
sim = "40/"
solution_policy = c("monolithic", "richardson")
plot_mesh(data_dir, mesh_plot_name, n_obs, sim, solution_policy)

## 20% NA, mesh: unit_square_coarse, obs:[500, 1000, 2000, 4000, 8000]
test_name = "boxplot_na_02_unit_square_coarse"
draw_boxplots_na_fixed(test_name, mesh_id, test_id, na_perc = 0.2)
mesh_plot_name = "diff-mesh-comparison-na-02.png"
data_dir = paste0("//wsl.localhost/Ubuntu/root/fdaPDE-cpp/test/data/models/mixed_srpde/", mesh_id,"/", test_id, "/")
n_obs = "500/"
sim = "40/"
solution_policy = c("monolithic", "richardson")
plot_mesh(data_dir, mesh_plot_name, n_obs, sim, solution_policy)

## Comparison between different NA percentages, NA:[0,0.05,0.1,0.15,0.2], mesh: unit_square_coarse, obs:2000
test_name = "boxplot_obs_2000_unit_square_coarse"
draw_boxplots_obs_fixed(test_name, mesh_id, test_id, n_obs=2000)


##### Mesh comparison

plot_boxplot_levels_mesh = function(data, m, method, xlabel_name = "levels",
                               filename="boxplot.pdf"){
  m_ = levels(data[[m]])
  methods_ = levels(data[[method]])
  at_ <- c()
  for(i in 1:length(m_)){
    at_ <-  c(at_, ((i-1)*(1+length(methods_)) + (1:length(methods_))))
  }
  
  fill_col = viridis::viridis((length(levels(data[[method]]))+1), begin=0.25, end=0.95)
  fill_col = fill_col[1:length(methods_)]
  facs = names(Filter(is.factor, data))
  which( ! names(data) %in% facs )
  doplot = names(data)[which( ! names(data) %in% facs )]
  
  if(length(methods_)%%2 != 0){
    at_label = seq(ceiling(length(methods)/2), at_[length(at_)], by=(length(methods_) + 1))
  }else{
    at_label = seq(length(methods_)/2, at_[length(at_)], by=(length(methods_) + 1))
  }
  
  pdf(filename, family = "serif", width = 7, height = 7)
  for(i in doplot){
    boxplot(data[[i]] ~  data[[method]] + as.numeric(data[[m]]),
            ylab="", xlab=xlabel_name, at = at_, xaxt="n",
            ylim=c(min(data[[i]]), max(data[[i]])*(1.1)),
            col=fill_col,cex.lab = 2, cex.axis = 2, cex.main = 2,
            main = i)
    axis(side = 1, at = at_label, labels = m_, cex.lab = 2, cex.axis = 1)
    legend("topleft",legend=methods_, fill=fill_col, cex=1.5, inset=0.0125, 
           bty="n")
    
  }
  
  for(k in 1:length(methods_)){
    for(i in doplot){
      tmp <- data[which(data[[method]] == methods_[k]),]
      boxplot(tmp[[i]] ~ tmp[[m]],
              ylab="", xlab="observations", xaxt="n",
              col=fill_col[k],cex.lab = 2, cex.axis = 2, cex.main = 2,
              main = paste0(i," (",methods_[k],")"))
      axis(side = 1, at = 1:length(m_), labels = m_, cex.lab = 2, cex.axis = 2)
    }
  }
  dev.off()
}

test_name = "mesh_comparison_01_na-8000_obs-with-gmres"
na_perc = 0.1
n_obs = 8000


mesh_ids = c("unit_square", "unit_square_coarse", "unit_square_medium")
test_id = "diff_locations_diff_NA"

imgdir = "imgs/"
if(!dir.exists(imgdir)) dir.create(imgdir)

imgdir = paste0(imgdir, test_id, "/")
if(!dir.exists(imgdir)) dir.create(imgdir)

results = list()  
for (mesh_id in mesh_ids) { 
  data_dir = paste0("//wsl.localhost/Ubuntu/root/fdaPDE-cpp/test/data/models/mixed_srpde/", mesh_id, "/", test_id, "/")
  mono = read.table(paste0(data_dir, "output/monolithic.txt"), header = TRUE) 
  rich = read.table(paste0(data_dir, "output/richardson.txt"), header = TRUE)
  gmres = read.table(paste0(data_dir, "output/richardson_gmres.txt"), header = TRUE)
  mono$solution_policy = rep("monolithic", times = nrow(mono))
  rich$solution_policy = rep("richardson", times = nrow(rich))
  gmres$solution_policy = rep("richardson_gmres", times = nrow(gmres))
  results[[length(results) + 1]] = list(mono, rich, gmres)  
}
results_df = do.call(rbind, unlist(results, recursive = FALSE))  
results_df$mesh = rep(mesh_ids, times = sapply(results, function(x) sum(sapply(x, nrow))))
# results = results_df[results_df$rmse_alpha < 1,]

results = results_df[results_df$na_perc == na_perc, ]
results = results[results$n_obs == n_obs,]

results$solution_policy = as.factor(results$solution_policy)
results$mesh = factor(results$mesh, levels=c("unit_square_coarse", "unit_square_medium", "unit_square"), ordered = TRUE)

# plots 
filename = paste0(test_name,".pdf")
plot_boxplot_levels_mesh(results, m="mesh", method="solution_policy", xlabel_name="Mesh ID",
                    filename = paste0(paste0(imgdir,filename)))



# ---------------------------- diff m
test_id = "same_locs_diff_moutput"
mesh_id = "unit_square_coarse"
data_dir = paste0("//wsl.localhost/Ubuntu/root/fdaPDE-cpp/test/data/models/mixed_srpde/", mesh_id, "/", test_id, "/")
mono = read.table(paste0(data_dir, "monolithic_gen.txt"), header = TRUE) 
rich = read.table(paste0(data_dir, "richardson_gen.txt"), header = TRUE)
gmres = read.table(paste0(data_dir, "richardson_gmres_gen.txt"), header = TRUE)
mono$solution_policy = rep("monolithic", times = nrow(mono))
rich$solution_policy = rep("richardson", times = nrow(rich))
gmres$solution_policy = rep("richardson_gmres", times = nrow(gmres))
results = list()  
results[[length(results) + 1]] = list(mono, rich, gmres)  
results_df = do.call(rbind, unlist(results, recursive = FALSE))  
# results_df$mesh = rep(mesh_id, times = nrow(results_df))
n_obs = 2000
results = results_df[results_df$n_obs == n_obs,]
filename = paste0(test_id,".pdf")

imgdir = "imgs/"
if(!dir.exists(imgdir)) dir.create(imgdir)

imgdir = paste0(imgdir, test_id, "/")
if(!dir.exists(imgdir)) dir.create(imgdir)

results$solution_policy = as.factor(results$solution_policy)
results$m = as.factor(results$m)

# plots 
filename = paste0(test_id,".pdf")
plot_boxplot_levels(results, m="m", method="solution_policy", xlabel_name="Number of levels",
                    filename = paste0(paste0(imgdir,filename)))

# without gmres
results = results[results$solution_policy != "richardson_gmres",]
results$solution_policy = factor(results$solution_policy, levels=c("monolithic","richardson"))
filename = paste0(test_id,"_without_gmres.pdf")
plot_boxplot_levels(results, m="m", method="solution_policy", xlabel_name="Number of levels",
                    filename = paste0(paste0(imgdir,filename)))


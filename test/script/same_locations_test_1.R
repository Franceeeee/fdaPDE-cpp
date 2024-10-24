# Test 1 -----------------------------------------------------------------------
if(!require(fdaPDEmixed)){
  devtools::install_github(repo ="aldoclemente/fdaPDEmixed")
}

# if(.Platform$GUI == "RStudio"){
#   if(!require(rstudioapi)) install.packages("rstudioapi")
# }
# ------------------------------------------------------------------------------
rm(list=ls())
library(fdaPDEmixed)
# !!! path/to/graphic-tools ----------------------------------------------------
path_ = "~/Desktop/graphic-tools/"
source(paste0(path_, "utils.R"))
ll <- parse(file = paste0(path_, "plot_smooth_2D.R"))

for (i in seq_along(ll)) {
  tryCatch(eval(ll[[i]]), 
           error = function(e) message(as.character(e)))
}
# ------------------------------------------------------------------------------

mesh_id = "unit_square_coarse"
test_id = "same_locations_test_1"
solution_policy = c("monolithic", "richardson")
data_dir = paste0("../data/models/mixed_srpde/", mesh_id,"/", test_id, "/")
ids = unlist(strsplit(data_dir, split="/"))

#mesh_id = ids[length(ids)-1]
#test_id = ids[length(ids)]

nodes = read.csv(paste0("../data/mesh/", mesh_id, "/points.csv"))[,2:3]
triangles = read.csv(paste0("../data/mesh/", mesh_id, "/elements.csv"))[,2:4]

mesh=create.mesh.2D(nodes=nodes, triangles = triangles)
plot(mesh)
FEMbasis <- create.FEM.basis(mesh)

# nice plots
n_obs = "1000/"
sim = "0/"

input_dir = paste0(data_dir, "input/", n_obs,sim)

locs = as.matrix(read.table(paste0(input_dir,"locs_0.txt"), header = F))
design_matrix = matrix(nrow=0, ncol=2)
for( i in 0:2){
  design_matrix = rbind(design_matrix, as.matrix(read.table(paste0(input_dir,"DesignMatrix_", i,".txt"), header = F)))
}

obs = matrix(nrow=nrow(locs),ncol=0)
for( i in 0:2){
  obs = cbind(obs, as.matrix(read.table(paste0(input_dir,"obs_", i,".txt"), header = F)))
}

# {
# x11()
# plot(mesh, pch=".")
# points(locs, pch=16)
# }

lambda = 1e-3
# fdaPDE ---------------------------------------------------------------------
output_mono <- fdaPDEmixed::smooth.FEM.mixed(observations = obs, locations = locs,
                                               covariates = design_matrix, random_effect = c(1),
                                               FEMbasis = FEMbasis, lambda = lambda,
                                               FLAG_ITERATIVE = FALSE)

#plot(FEM(output_mono$fit.FEM.mixed$coeff[1:nrow(mesh$nodes)], FEMbasis))
#plot(FEM(output_mono$fit.FEM.mixed$coeff[(nrow(mesh$nodes)+1):(2*nrow(mesh$nodes))], FEMbasis))
#plot(FEM(output_mono$fit.FEM.mixed$coeff[(2*nrow(mesh$nodes)+1):(3*nrow(mesh$nodes))], FEMbasis))

output_mono$beta
output_mono$b_i
output_mono_dir = paste0(data_dir,"output/",n_obs, sim,solution_policy[1],"/")
output_mono_cpp = list(coeff = as.matrix(read.table(paste0(output_mono_dir, "estimate_f.txt"), header = F)),
                  beta = as.matrix(read.table(paste0(output_mono_dir,"beta.txt"), header = F)),
                  alpha_i = as.matrix(read.table(paste0(output_mono_dir,"alpha.txt"), header = F)),
                  f_0 = as.matrix(read.table(paste0(output_mono_dir,"estimate_f_0.txt"), header = F)),
                  f_1 = as.matrix(read.table(paste0(output_mono_dir,"estimate_f_1.txt"), header = F)),
                  f_2 = as.matrix(read.table(paste0(output_mono_dir,"estimate_f_2.txt"), header = F)))

max(abs(output_mono$fit.FEM.mixed$coeff - output_mono_cpp$coeff))
max(abs(output_mono$beta - output_mono_cpp$beta))
max(abs(output_mono$b_i - output_mono_cpp$alpha_i))

output_iter <- fdaPDEmixed::smooth.FEM.mixed(observations = obs, locations = locs,
                                               covariates = design_matrix, random_effect = c(1),
                                               FEMbasis = FEMbasis, lambda = lambda,
                                               FLAG_ITERATIVE = TRUE, anderson_memory = 1L)

output_iter$iterations
output_iter$beta
output_iter$b_i

output_iter_dir = paste0(data_dir,"output/",n_obs, sim,solution_policy[2],"/")
output_iter_cpp = list(coeff = as.matrix(read.table(paste0(output_iter_dir, "estimate_f.txt"), header = F)),
                       beta = as.matrix(read.table(paste0(output_iter_dir,"beta.txt"), header = F)),
                       alpha_i = as.matrix(read.table(paste0(output_iter_dir,"alpha.txt"), header = F)),
                       f_0 = as.matrix(read.table(paste0(output_iter_dir,"estimate_f_0.txt"), header = F)),
                       f_1 = as.matrix(read.table(paste0(output_iter_dir,"estimate_f_1.txt"), header = F)),
                       f_2 = as.matrix(read.table(paste0(output_iter_dir,"estimate_f_2.txt"), header = F)))

max(abs(output_iter$fit.FEM.mixed$coeff - output_iter_cpp$coeff))
max(abs(output_iter$beta - output_iter_cpp$beta))
max(abs(output_iter$b_i - output_iter_cpp$alpha_i))

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

# named list ...
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
plot_smooth_2D(FEM(as.matrix(read.table(paste0(data_dir,"input/cov_1.txt"),header = F)),FEMbasis), 
               colorscale = viridis)
snapshot3d(filename = paste0(imgdir, "cov1.png"),
           fmt = "png", width = 800, height = 750, webshot = rgl.useNULL())
close3d()

plot_colorbar(f_0, coeff_lims = coeff_lims_0, colorscale = viridis,
              file = paste0(imgdir, "colorbar_cov1"))

}

# ------------------------------------------------------------------------------
mono = read.table(paste0(data_dir,"output/monolithic.txt"), header = T) 
rich = read.table(paste0(data_dir,"output/richardson.txt"), header = T)
mono$solution_policy = rep("monolithic", times = nrow(mono))
rich$solution_policy = rep("richardson", times = nrow(rich))

#
results = rbind(mono, rich)
results$solution_policy = as.factor(results$solution_policy)
results$n_obs = as.factor(results$n_obs)

# plots 
plot_boxplot(results, n_obs="n_obs", method="solution_policy", 
             filename = paste0(paste0(imgdir,"boxplots.pdf")))

{ 
  at_ <- c(1:2, 4:5, 7:8, 10:11, 13:14)
  fill_col <- viridis::viridis(3, begin=0.25, end=0.95)
  fill_col <- fill_col[1:2]
  legend <- levels(results$solution_policy)
  
  n_obs = as.integer(levels(results$n_obs))
  mar_ = par("mar")
  mar_[2] = mar_[2] + 0.25
  pdf(paste0(imgdir, "time.pdf"), family = "serif", width = 7, height = 7)
  
  monolithic <- results[results$solution_policy == "monolithic", ]
  iterative <- results[results$solution_policy == "iterative", ]
  
  pdf(paste0(imgdir, "time_mean.pdf"), family = "serif", width = 7, height = 7)
  plot(log(n_obs), log(tapply(monolithic$time, monolithic$n_obs, mean)),
       type="l", lty=2, lwd=4, col=fill_col[2], 
       ylim= c(min(log(results$time)), max(log(results$time))),
       ylab="", xlab="log(nodes)", xaxt="n",
       cex.lab = 2, cex.axis = 2, cex.main = 2,
       main ="init time [s]")
  points(log(n_obs), log(tapply(monolithic$time, monolithic$n_obs, mean)),
         pch=16, cex=2, col=fill_col[2])
  points(log(n_obs), log(tapply(iterative$time, iterative$n_obs, mean)),
         type="l", lty=2, lwd=4, col=fill_col[1])
  points(log(n_obs), log(tapply(iterative$time, iterative$n_obs, mean)),
         pch=16, cex=2, col=fill_col[1])
  # points(log(n_obs), log(n_obs)-10,
  #        type="l", lty=3, lwd=3, col="black")
  # points(log(n_obs), 2*log(n_obs)-16,
  #        type="l", lty=3, lwd=3, col="red")
  axis(side = 1, at = log(n_obs), labels = round(log(n_obs), digits = 2), cex.lab = 2, cex.axis = 2)
  legend("topleft", legend=legend, fill=fill_col, horiz=F, cex=1.5, inset=0.0125, 
         bty="n")
  
  plot(n_obs, tapply(monolithic$time, monolithic$n_obs, mean),
       type="l", lty=2, lwd=4, col=fill_col[2], 
       ylim= c(min(results$time), max(results$time)),
       ylab="", xlab="observations", xaxt="n",
       cex.lab = 2, cex.axis = 2, cex.main = 2,
       main ="init time [s]")
  points(n_obs, tapply(monolithic$time, monolithic$n_obs, mean),
         pch=16, cex=2, col=fill_col[2])
  points(n_obs, tapply(iterative$time, iterative$n_obs, mean),
         type="l", lty=2, lwd=4, col=fill_col[1])
  points(n_obs, tapply(iterative$time, iterative$n_obs, mean),
         pch=16, cex=2, col=fill_col[1])
  axis(side = 1, at = n_obs, labels = n_obs, cex.lab = 2, cex.axis = 2)
  legend("topleft", legend=legend, fill=fill_col, horiz=F, cex=1.5, inset=0.0125, 
         bty="n")
  dev.off()
  
  pdf(paste0(imgdir, "init_time_mean.pdf"), family = "serif", width = 7, height = 7)
  plot(log(n_obs), log(tapply(monolithic$time_init, monolithic$n_obs, mean)),
       type="l", lty=2, lwd=4, col=fill_col[2], 
       ylim= c(min(log(results$time_init)), max(log(results$time_init))),
       ylab="", xlab="log(nodes)", xaxt="n",
       cex.lab = 2, cex.axis = 2, cex.main = 2,
       main ="init time [s]")
  points(log(n_obs), log(tapply(monolithic$time_init, monolithic$n_obs, mean)),
         pch=16, cex=2, col=fill_col[2])
  points(log(n_obs), log(tapply(iterative$time_init, iterative$n_obs, mean)),
         type="l", lty=2, lwd=4, col=fill_col[1])
  points(log(n_obs), log(tapply(iterative$time_init, iterative$n_obs, mean)),
         pch=16, cex=2, col=fill_col[1])
  # points(log(n_obs), log(n_obs)-10,
  #        type="l", lty=3, lwd=3, col="black")
  # points(log(n_obs), 2*log(n_obs)-16,
  #        type="l", lty=3, lwd=3, col="red")
  axis(side = 1, at = log(n_obs), labels = round(log(n_obs), digits = 2), cex.lab = 2, cex.axis = 2)
  legend("topleft", legend=legend, fill=fill_col, horiz=F, cex=1.5, inset=0.0125, 
         bty="n")
  
  plot(n_obs, tapply(monolithic$time_init, monolithic$n_obs, mean),
       type="l", lty=2, lwd=4, col=fill_col[2], 
       ylim= c(min(results$time_init), max(results$time_init)),
       ylab="", xlab="observations", xaxt="n",
       cex.lab = 2, cex.axis = 2, cex.main = 2,
       main ="init time [s]")
  points(n_obs, tapply(monolithic$time_init, monolithic$n_obs, mean),
         pch=16, cex=2, col=fill_col[2])
  points(n_obs, tapply(iterative$time_init, iterative$n_obs, mean),
         type="l", lty=2, lwd=4, col=fill_col[1])
  points(n_obs, tapply(iterative$time_init, iterative$n_obs, mean),
         pch=16, cex=2, col=fill_col[1])
  axis(side = 1, at = n_obs, labels = n_obs, cex.lab = 2, cex.axis = 2)
  legend("topleft", legend=legend, fill=fill_col, horiz=F, cex=1.5, inset=0.0125, 
         bty="n")
  dev.off()
  
  pdf(paste0(imgdir, "init_time_median.pdf"), family = "serif", width = 7, height = 7)
  plot(log(n_obs), log(tapply(monolithic$time_init, monolithic$n_obs, median)),
       type="l", lty=2, lwd=4, col=fill_col[2], 
       ylim= c(min(log(results$time_init)), max(log(results$time_init))),
       ylab="", xlab="log(nodes)", xaxt="n",
       cex.lab = 2, cex.axis = 2, cex.main = 2,
       main ="init time [s]")
  points(log(n_obs), log(tapply(monolithic$time_init, monolithic$n_obs, median)),
         pch=16, cex=2, col=fill_col[2])
  points(log(n_obs), log(tapply(iterative$time_init, iterative$n_obs, median)),
         type="l", lty=2, lwd=4, col=fill_col[1])
  points(log(n_obs), log(tapply(iterative$time_init, iterative$n_obs, median)),
         pch=16, cex=2, col=fill_col[1])
  # points(log(n_obs), log(n_obs)-10,
  #        type="l", lty=3, lwd=3, col="black")
  # points(log(n_obs), 2*log(n_obs)-16,
  #        type="l", lty=3, lwd=3, col="red")
  axis(side = 1, at = log(n_obs), labels = round(log(n_obs), digits = 2), cex.lab = 2, cex.axis = 2)
  legend("topleft", legend=legend, fill=fill_col, horiz=F, cex=1.5, inset=0.0125, 
         bty="n")
  
  plot(n_obs, tapply(monolithic$time_init, monolithic$n_obs, median),
       type="l", lty=2, lwd=4, col=fill_col[2], 
       ylim= c(min(results$time_init), max(results$time_init)),
       ylab="", xlab="observations", xaxt="n",
       cex.lab = 2, cex.axis = 2, cex.main = 2,
       main ="init time [s]")
  points(n_obs, tapply(monolithic$time_init, monolithic$n_obs, median),
         pch=16, cex=2, col=fill_col[2])
  points(n_obs, tapply(iterative$time_init, iterative$n_obs, median),
         type="l", lty=2, lwd=4, col=fill_col[1])
  points(n_obs, tapply(iterative$time_init, iterative$n_obs, median),
         pch=16, cex=2, col=fill_col[1])
  axis(side = 1, at = n_obs, labels = n_obs, cex.lab = 2, cex.axis = 2)
  legend("topleft", legend=legend, fill=fill_col, horiz=F, cex=1.5, inset=0.0125, 
         bty="n")
  dev.off()
  
  
  pdf(paste0(imgdir, "solve_time_mean.pdf"), family = "serif", width = 7, height = 7)
  plot(log(n_obs), log(tapply(monolithic$time_solve, monolithic$n_obs, mean)),
       type="l", lty=2, lwd=4, col=fill_col[2], 
       ylim= c(min(log(results$time_solve)), max(log(results$time_solve))),
       ylab="", xlab="log(nodes)", xaxt="n",
       cex.lab = 2, cex.axis = 2, cex.main = 2,
       main ="init time [s]")
  points(log(n_obs), log(tapply(monolithic$time_solve, monolithic$n_obs, mean)),
         pch=16, cex=2, col=fill_col[2])
  points(log(n_obs), log(tapply(iterative$time_solve, iterative$n_obs, mean)),
         type="l", lty=2, lwd=4, col=fill_col[1])
  points(log(n_obs), log(tapply(iterative$time_solve, iterative$n_obs, mean)),
         pch=16, cex=2, col=fill_col[1])
  # points(log(n_obs), log(n_obs)-10,
  #        type="l", lty=3, lwd=3, col="black")
  # points(log(n_obs), 2*log(n_obs)-16,
  #        type="l", lty=3, lwd=3, col="red")
  axis(side = 1, at = log(n_obs), labels = round(log(n_obs), digits = 2), cex.lab = 2, cex.axis = 2)
  legend("topleft", legend=legend, fill=fill_col, horiz=F, cex=1.5, inset=0.0125, 
         bty="n")
  
  plot(n_obs, tapply(monolithic$time_solve, monolithic$n_obs, mean),
       type="l", lty=2, lwd=4, col=fill_col[2], 
       ylim= c(min(results$time_solve), max(results$time_solve)),
       ylab="", xlab="observations", xaxt="n",
       cex.lab = 2, cex.axis = 2, cex.main = 2,
       main ="solve time [s]")
  points(n_obs, tapply(monolithic$time_solve, monolithic$n_obs, mean),
         pch=16, cex=2, col=fill_col[2])
  points(n_obs, tapply(iterative$time_solve, iterative$n_obs, mean),
         type="l", lty=2, lwd=4, col=fill_col[1])
  points(n_obs, tapply(iterative$time_solve, iterative$n_obs, mean),
         pch=16, cex=2, col=fill_col[1])
  axis(side = 1, at = n_obs, labels = n_obs, cex.lab = 2, cex.axis = 2)
  legend("topleft", legend=legend, fill=fill_col, horiz=F, cex=1.5, inset=0.0125, 
         bty="n")
  dev.off()
  
  pdf(paste0(imgdir, "solve_time_median.pdf"), family = "serif", width = 7, height = 7)
  plot(log(n_obs), log(tapply(monolithic$time_solve, monolithic$n_obs, median)),
       type="l", lty=2, lwd=4, col=fill_col[2], 
       ylim= c(min(log(results$time_solve)), max(log(results$time_solve))),
       ylab="", xlab="log(nodes)", xaxt="n",
       cex.lab = 2, cex.axis = 2, cex.main = 2,
       main ="init time [s]")
  points(log(n_obs), log(tapply(monolithic$time_solve, monolithic$n_obs, median)),
         pch=16, cex=2, col=fill_col[2])
  points(log(n_obs), log(tapply(iterative$time_solve, iterative$n_obs, median)),
         type="l", lty=2, lwd=4, col=fill_col[1])
  points(log(n_obs), log(tapply(iterative$time_solve, iterative$n_obs, median)),
         pch=16, cex=2, col=fill_col[1])
  # points(log(n_obs), log(n_obs)-10,
  #        type="l", lty=3, lwd=3, col="black")
  # points(log(n_obs), 2*log(n_obs)-16,
  #        type="l", lty=3, lwd=3, col="red")
  axis(side = 1, at = log(n_obs), labels = round(log(n_obs), digits = 2), cex.lab = 2, cex.axis = 2)
  legend("topleft", legend=legend, fill=fill_col, horiz=F, cex=1.5, inset=0.0125, 
         bty="n")
  
  plot(n_obs, tapply(monolithic$time_solve, monolithic$n_obs, median),
       type="l", lty=2, lwd=4, col=fill_col[2], 
       ylim= c(min(results$time_solve), max(results$time_solve)),
       ylab="", xlab="observations", xaxt="n",
       cex.lab = 2, cex.axis = 2, cex.main = 2,
       main ="solve time [s]")
  points(n_obs, tapply(monolithic$time_solve, monolithic$n_obs, median),
         pch=16, cex=2, col=fill_col[2])
  points(n_obs, tapply(iterative$time_solve, iterative$n_obs, median),
         type="l", lty=2, lwd=4, col=fill_col[1])
  points(n_obs, tapply(iterative$time_solve, iterative$n_obs, median),
         pch=16, cex=2, col=fill_col[1])
  axis(side = 1, at = n_obs, labels = n_obs, cex.lab = 2, cex.axis = 2)
  legend("topleft", legend=legend, fill=fill_col, horiz=F, cex=1.5, inset=0.0125, 
         bty="n")
  dev.off()
  
}



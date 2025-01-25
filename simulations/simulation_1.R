
library(Matrix)
library(ggplot2)
library(viridis)

dofs <- as.matrix(readMM("simulation_1_dofs.mtx"))
f <- as.matrix(readMM("simulation_1_f.mtx"))
# f_exact =  as.matrix(readMM("simulation_1_f_exact.mtx"))

mycolors<- function(x) {
    colors<-viridis( x + 1 )
    colors[1:x]
}

n_breaks <- 50
# qua andrebbe considerata anche la "exact" per fare grafici coerenti (?)
mybreaks <- seq(min(f), max(f), length.out = n_breaks) 

data <- data.frame(x = round(dofs[,1], 10), y = round(dofs[,2], 10), z = as.matrix(f[,1]))    
plt <-
    ggplot(aes(x = x, y = y, z = z), data = data) +
    geom_contour_filled(breaks = mybreaks) +
    scale_fill_manual(
        aesthetics = "fill",
        values = mycolors(n_breaks + 2)
        ) +
    coord_fixed() +
    ## geom_contour(colour = "white", linejoin = "round") +
    ggtitle("Titolo ?") + 
    theme(legend.position = "none")
plt
ggsave("simulation_1_f.png")

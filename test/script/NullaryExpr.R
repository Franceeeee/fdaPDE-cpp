rm(list = ls())
U_ = read.table("../build/U.txt", header = F)

U_0 = read.table("../build/U_0.txt", header = F)
U_1 = read.table("../build/U_1.txt", header = F)
U_2 = read.table("../build/U_2.txt", header = F)

U_view_0 = read.table("../build/U_view_0.txt", header = F)
U_view_1 = read.table("../build/U_view_1.txt", header = F)
U_view_2 = read.table("../build/U_view_2.txt", header = F)

max(abs(U_[,1:2] - U_0[1:nrow(U_),1:2]))
max(abs(U_[,3:4] - U_1[1:nrow(U_),c(1,3)]))
max(abs(U_[,5:6] - U_2[1:nrow(U_),c(1,4)]))

max(abs(U_view_0[,1] - U_0[,1]))
max(abs(U_view_1[,1] - U_1[,1]))

max(abs(U_view_0 - U_0))
max(abs(U_view_1 - U_1))
max(abs(U_view_2 - U_2))

cbind(head(U_view_0), head(U_0))
cbind(head(U_view_1), head(U_1))
cbind(head(U_view_2), head(U_2))
cbind(tail(U_view_0), tail(U_0))
cbind(tail(U_view_1), tail(U_1))
cbind(tail(U_view_2), tail(U_2))

# # # 
#V_ = read.table("U.txt", header = F)
V_0 = read.table("../build/V_0.txt", header = F)
V_1 = read.table("../build/V_1.txt", header = F)
V_2 = read.table("../build/V_2.txt", header = F)

V_view_0 = read.table("../build/V_view_0.txt", header = F)
V_view_1 = read.table("../build/V_view_1.txt", header = F)
V_view_2 = read.table("../build/V_view_2.txt", header = F)

max(abs(V_view_0 - V_0))
max(abs(V_view_1 - V_1))
max(abs(V_view_2 - V_2))

rbind(V_view_0[1:4,1:6], V_0[1:4,1:6])
rbind(V_view_1[1:4,1:6], V_1[1:4,1:6])
rbind(V_view_2[1:4,1:6], V_2[1:4,1:6])

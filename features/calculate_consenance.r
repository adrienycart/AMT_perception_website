library(incon)

# args <- commandArgs(trailingOnly=TRUE)

# filename <- args[1]
# chord <- c()
# for (i in 2:length(args)) {
#     chord <- c(chord, as.integer(args[i]))
# }

models <- c("hutch_78_roughness", "har_18_harmonicity", "har_19_corpus")
consenance_values <- incon(chord, models)

saveRDS(consenance_values, file=paste("consenance_values/", filename, sep=""))


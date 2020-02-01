library(incon)

folders <- list.dirs('features/chords_and_times', recursive=FALSE)

systems <- c('kelz', 'lisu', 'google', 'cheng')
models <- c("hutch_78_roughness", "har_18_harmonicity", "har_19_corpus")

for (folder in folders) {
    example <- substring(folder, 27)
    print(example)
    for (system in systems) {
        
        
    }
}


# consenance_values <- incon(chord, models)

# saveRDS(consenance_values, file=paste("consenance_values/", filename, sep=""))


#!/usr/bin/env Rscript
# TRUE multinomial MD-GAM (Ghosh, SahaRay & Sarkar 2025): a single multinomial GAM
# additive in the per-class distances, NOT one-vs-rest. mgcv's multinom family with
# penalized (REML/newton) smoothing is numerically unstable on separable classes, so
# we use FIXED-df regression splines (fx=TRUE): a genuine additive-spline multinomial
# GAM whose per-smooth df is preset, sidestepping the outer smoothing optimization.
# Usage: Rscript mgcv_mdgam_multinom.R train.csv test.csv out.csv
suppressMessages(library(mgcv))
a <- commandArgs(trailingOnly = TRUE)
tr <- read.csv(a[1]); te <- read.csv(a[2])
dcols <- grep("^d", names(tr), value = TRUE)
J <- length(unique(tr$y))
rhs <- paste(sapply(dcols, function(c) sprintf("s(%s, k=6, fx=TRUE)", c)), collapse = " + ")
flist <- c(list(as.formula(paste("y ~", rhs))),
           replicate(J - 2, as.formula(paste("~", rhs)), simplify = FALSE))
fit <- gam(flist, data = tr, family = multinom(K = J - 1))
P <- predict(fit, newdata = te, type = "response")   # n x J probability matrix
write.csv(data.frame(pred = max.col(P) - 1L), a[3], row.names = FALSE)

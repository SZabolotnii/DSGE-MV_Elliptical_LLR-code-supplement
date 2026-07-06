#!/usr/bin/env Rscript
# Faithful global MD-GAM head of Ghosh, SahaRay & Sarkar (JMVA 2025, arXiv 2402.08283):
# per-class UNSQUARED Mahalanobis distances -> penalized-spline GAM, logistic link, REML
# smoothing (Wood 2017 / mgcv). BINARY fit; multiclass is handled one-vs-rest by the
# Python driver (mgcv multinom+REML is numerically unstable on separable classes).
# Usage: Rscript mgcv_mdgam.R train.csv test.csv out.csv
# train.csv: columns d0..d{J-1}, y in {0,1} ; test.csv: d0..d{J-1}.
# Writes both the fitted probability (prob) and the hard 0/1 label (pred).
suppressMessages(library(mgcv))
a <- commandArgs(trailingOnly = TRUE)
tr <- read.csv(a[1]); te <- read.csv(a[2])
dcols <- grep("^d", names(tr), value = TRUE)
rhs <- paste(sapply(dcols, function(c) {
  ku <- min(10, length(unique(tr[[c]])) - 1); sprintf("s(%s, k=%d)", c, max(3, ku))
}), collapse = " + ")
fit <- gam(as.formula(paste("y ~", rhs)), data = tr, family = binomial, method = "REML")
prob <- as.numeric(predict(fit, newdata = te, type = "response"))
write.csv(data.frame(prob = prob, pred = as.integer(prob > 0.5)), a[3], row.names = FALSE)

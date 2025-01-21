## LOO MSE

model_list <- c('LR', 'Ridge', 'LASSO', 'KNN', 'ClassTree')

# Limit to 50 iterations because of computational concerns
B <- 50

mse_collection <- rep(0,5)

df <- heart

# place to store the bootstrap samples
boot_samples <- vector(mode = "list", length = B)

# place to store the misclassifications for each index 
#loo_errors <- rep(0, nrow(df))

loo_errors <- matrix(0, nrow = 5, ncol = nrow(df))
rownames(loo_errors) <- model_list

# place to store the number of bootstrap samples that contains each index
C_vec <- rep(0, nrow(df))

gridd <- exp(seq(2,-6, -0.5))

set.seed(1994)

# carrying out the bootstrap
for(i in 1:B) {
  
  # resampling the indices of the original dataframe
  # do this without replacement
  boot_idx <- sort(sample(nrow(df),nrow(df),replace=TRUE))
  boot_samples[[i]] <- boot_idx
}

# calculating the errors per index
for (i in 1:nrow(df)) {
  for (j in 1:B) {
    # if the boot sample does not contain the index, fit a knn model
    if (!(i %in% boot_samples[[j]])) {
      trn <- df[boot_samples[[j]],]
      tst <- df[i,]
      cls <- df$DEATH_EVENT[boot_samples[[j]]]
      
      # model 1
      
      lr_1 <-  glm(fmla,data = trn, family=binomial)
      lr_mod <- step(lr_1, direction = 'backward', trace=0)
      
      # model 2
      cv.rdg.fit <- cv.glmnet(as.matrix(trn[,xnams]), trn[,'DEATH_EVENT'], family = 'binomial', alpha = 0, lambda = gridd, nfolds = 10, type.measure = 'class')
      
      # model 3
      cv.lso.fit <- cv.glmnet(as.matrix(trn[,xnams]), trn[,'DEATH_EVENT'], family = 'binomial', alpha = 1, lambda = gridd, nfolds = 10, type.measure = 'class')
      
      # model 4
      knn_cv_res <- knn.cv(x=as.matrix(trn[,xnams]), y=trn$DEATH_EVENT, nfolds = 10, stratified = FALSE, k=1:10, type = 'C')
      knn_mod <- class::knn(trn[,-ncol(trn)], tst[,-ncol(tst)], cls, k=which.max(knn_cv_res$crit))
      
      # model 5
      t1_mod <- tree(fmla, trn)
      cv.heart_mod <- cv.tree(t1_mod,FUN = prune.misclass)
      classtree <- prune.tree(t1_mod,best=cv.heart_mod$size[which.min(cv.heart_mod$dev)])
      
      
      # check for misclassification and update the loo_error
      loo_errors[1,i] <- loo_errors[1,i] + as.numeric(ifelse(predict(lr_mod, newdata = tst, type = 'response') > 0.5,1,0) != df$DEATH_EVENT[i])
      
      loo_errors[2,i] <- loo_errors[2,i] + as.numeric(ifelse(predict(cv.rdg.fit, s=cv.rdg.fit$lambda.min, newx = as.matrix(tst[,xnams]), type = 'response') > 0.5,1,0) != df$DEATH_EVENT[i])
      
      loo_errors[3,i] <- loo_errors[3,i] + as.numeric(ifelse(predict(cv.lso.fit, s=cv.lso.fit$lambda.min, newx = as.matrix(tst[,xnams]), type = 'response') > 0.5,1,0) != df$DEATH_EVENT[i])
      
      loo_errors[4,i] <- loo_errors[4,i] + as.numeric(knn_mod != df$DEATH_EVENT[i])
      
      probs <- predict(classtree, tst)
      yhat <- ifelse(probs[,1] > 0.5, 0, 1)
      
      loo_errors[5,i] <- loo_errors[5,i] + as.numeric(yhat != df$DEATH_EVENT[i])
      
      
      # track that this bootstrap sample contained the index
      C_vec[i] <- C_vec[i] + 1
    }
  }
}

# finding the average error rate among the results

error_rates <- apply(loo_errors, 1, function(x){x/C_vec})

unique(error_rates)[,1]
dim(error_rates)
View(error_rates)

error_df <- apply(error_rates, 2, mean)
View(as.data.frame(error_df))

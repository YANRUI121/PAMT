library(coin)
library(survival)
library(survminer)
library(writexl)


setwd("C:\\Users\\yanrui\\Desktop\\COXPAMT\\LUSC")


##################################

select_by_p = read.csv("C:\\Users\\yanrui\\Desktop\\COXPAMT\\LUSC\\survival_dataset_test.csv")

select_by_p$risk_score_scale <- scale(select_by_p$risk_score)

select_by_p$Risk <- ifelse(select_by_p$risk_score_scale > median(select_by_p$risk_score_scale),"High","Low")
head(select_by_p)

write_xlsx(select_by_p,"C:\\Users\\yanrui\\Desktop\\COXPAMT\\LUSC\\survival_dataset_test.xlsx")


#write.csv(select_by_p[,1:3 ],"survival_data_20230430.csv",row.names = TRUE)

##################################

fit_cox <- coxph(Surv(survival_time_day, censor_state)~ risk_score_scale, data = select_by_p)
summary(fit_cox)
#ggforest(fit_cox,main="hazard ratio",cpositions=c(0.02,0.22,0.4),fontsize=0.8,refLabel="reference",noDigits=2,data = select_by_p)



fit <- survfit(Surv(survival_time_day, censor_state) ~ select_by_p$Risk, data = select_by_p)


dsurvplot <- ggsurvplot_list(fit,data = select_by_p,pval = T,
                     conf.int = F,
                     risk.table = T, 
                     risk.table.col = "strata", 
                     ###linetype = "strata",
                     surv.median.line = "hv", 
                     risk.table.y.text.col = F,risk.table.y.text = T,
                     legend.labs = c("H", "L"),
                     ggtheme = theme_bw()+theme(legend.text = element_text(colour = c("red", "blue")))
                     +theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())
                     +theme(plot.title = element_text(hjust = 0.5,size = 12,face = "bold"),axis.title.y.left = element_text(size = 16,face = "bold",vjust = 1),axis.title.x.bottom = element_text(size = 16,face = "bold",vjust = 0))
                     +theme(axis.text.x.bottom = element_text(size = 12,face = "bold",vjust = -0.8,colour = "black"))
                     +theme(axis.text.y.left = element_text(size = 12,face = "bold",vjust = 0.5,hjust = 0.5,angle = 90,colour = "black"))
                     +theme(legend.title = element_text(face = "bold",family = "Times",colour = "black",size = 12))
                     +theme(legend.text = element_text(face = "bold",family = "Times",colour = "black",size =12)), # Change ggplot2 theme
                     palette = c("red", "blue"),
                     xlim=c(10,2700),
                     xlab = "Days")##or Months

tiff(file="Rplot_lusc_dataset.tiff",height=5,width=4,unit="in",res=300)
dsurvplot
dev.off()





#ggsurvplot(fit, data = select_by_p)



#####################timeROC##############################

#rm(list = ls())
library(timeROC)
library(survival)

#select_by_p = select_by_p_001

#dev.new()


ROC <- timeROC(T=select_by_p$survival_time_day,   
               delta=select_by_p$censor_state,   
               marker=select_by_p$risk_score_scale,   
               cause=1,               
               weighting="marginal",   
               times=c(365*1, 365*3, 365*5),      
               iid=TRUE)

pdf(file='ROC_final_survival_lusc_dataset.pdf',height=6,width=8, family='Times')
tiff(file="ROC_final_survival_lusc_dataset.tiff",height=6,width=8,unit="in",res=300)

plot(ROC, 
     time=365*1, col="red", lwd=3, title = "")  
plot(ROC,
     time=365*3, col="blue", add=TRUE, lwd=3)   
plot(ROC,
     time=365*5, col="orange", add=TRUE, lwd=3)


legend("bottomright",
       c(paste0("AUC at 1 year: ",sprintf("%0.3f", ROC[["AUC"]][1])),
         paste0("AUC at 3 year: ",sprintf("%0.3f", ROC[["AUC"]][2])),
         paste0("AUC at 5 year: ",sprintf("%0.3f", ROC[["AUC"]][3]))),
       col=c("red", "blue", "orange"),
       cex=2,
       lty=1, lwd=2,bty = "n")


dev.off()#close PDF


summary(fit_cox)


